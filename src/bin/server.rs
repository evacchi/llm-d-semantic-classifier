//! llm-d-sc service binary (Kubernetes container entrypoint).
//!
//! Binds the existing [`ClassifyServer`] on `LLM_D_SC_LISTEN` (default
//! `0.0.0.0:50051`) and reads the ModelCar mount directory from
//! `LLM_D_SC_MODEL_DIR` (default `/models`). The served pipeline is the RESIDENT
//! Candle classifier: the binary reads the ModelCar dir, validates its required
//! layout, loads tokenizer + config + safetensors, constructs the real
//! `CandleClassifier`, and runs a WARMUP FORWARD on a fixture input — only then
//! does it report READY. ANY failure leaves the service NOT ready with an
//! actionable typed error (a directory that merely exists never produces READY,
//! AC-002/AC-003). The deterministic synthetic pipeline is NOT used here; it is
//! reserved for weight-free tests.
//!
//! Graceful shutdown (U-035): on SIGTERM (what Kubernetes sends) or SIGINT the
//! service stops admission, lets in-flight classifications finish, and exits 0.
//! The drain is BOUNDED by `LLM_D_SC_SHUTDOWN_GRACE_SECS` (default 20s, i.e.
//! inside Kubernetes' default 30s `terminationGracePeriodSeconds`); whatever is
//! still open when the grace elapses is severed, and a SECOND signal during the
//! drain terminates immediately (default disposition).

use std::env;
use std::io;

use llm_d_sc::classify::load_and_warm_modelcar;
use llm_d_sc::grpc::classify::ClassifyServer;
use llm_d_sc::metrics::LatencyStage;

/// Default TCP listen address.
const DEFAULT_LISTEN: &str = "0.0.0.0:50051";
/// Default ModelCar mount directory.
const DEFAULT_MODEL_DIR: &str = "/models";
/// Environment variable bounding the graceful drain.
const SHUTDOWN_GRACE_ENV: &str = "LLM_D_SC_SHUTDOWN_GRACE_SECS";
/// Default drain bound.
///
/// A LATENCY-to-die choice: Kubernetes' default `terminationGracePeriodSeconds`
/// is 30s, and the drain must finish INSIDE it or the kubelet SIGKILLs the pod
/// mid-request. 20s leaves margin for the final metrics log and runtime
/// teardown while still bounding how long a stuck client connection can hold
/// the pod.
const DEFAULT_SHUTDOWN_GRACE_SECS: u64 = 20;

fn main() -> io::Result<()> {
    let listen = env::var("LLM_D_SC_LISTEN").unwrap_or_else(|_| DEFAULT_LISTEN.to_string());
    let model_dir =
        env::var("LLM_D_SC_MODEL_DIR").unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string());
    if model_dir.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "LLM_D_SC_MODEL_DIR must not be empty",
        ));
    }

    // Real model lifecycle: validate the ModelCar required-files layout, load
    // tokenizer + config + safetensors, build the Candle classifier, and run a
    // WARMUP FORWARD on a fixture input. ANY failure leaves the service NOT
    // ready with an actionable typed error — a directory that merely exists
    // must NOT produce READY (AC-002/AC-003).
    let classifier = load_and_warm_modelcar(&model_dir).map_err(|e| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("llm-d-sc NOT ready: {model_dir}: {e}"),
        )
    })?;

    // Only a loaded+warmed classifier reaches here, so the server reports READY.
    let server = ClassifyServer::bind_with_classifier(&listen, classifier)?;
    eprintln!(
        "llm-d-sc: bound {listen} -> {}; ModelCar dir {model_dir}; READY (resident Candle classifier loaded and warmed)",
        server.local_addr()
    );

    // Periodically log the per-stage latency DECOMPOSITION.
    //
    // S-080 requires system evidence that distinguishes round-trip time from
    // queue and forward time. RTT is measurable from outside by any client; the
    // internal stages are not, and there is no metrics endpoint yet (tracked for
    // 0.3). Logging percentiles, not means, keeps this consistent with the rule
    // that a latency claim from an average is not evidence. Emitted only when
    // requests have actually been served, so an idle service stays quiet.
    //
    // The loop wakes on `interval` OR on shutdown: `recv_timeout` returns
    // `Disconnected` as soon as main drops the stop sender, so the thread exits
    // promptly instead of sleeping through the drain (and the FINAL metrics
    // line is then logged by main, after the drain, where the totals are real).
    let metrics = server.metrics();
    let interval = env::var("LLM_D_SC_METRICS_LOG_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30);
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let logger_metrics = metrics.clone();
    let metrics_logger = std::thread::Builder::new()
        .name("metrics-log".to_string())
        .spawn(move || {
            let mut last_total = 0u64;
            loop {
                match stop_rx.recv_timeout(std::time::Duration::from_secs(interval)) {
                    // Interval elapsed: log the decomposition if anything was
                    // served since the last line.
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    // Main dropped the stop sender: shutdown began.
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) | Ok(()) => return,
                }
                let snap = logger_metrics.snapshot();
                let served = snap.cache_hits + snap.cache_misses + snap.cache_coalesced;
                if served == last_total {
                    continue;
                }
                last_total = served;
                let stage = |s| logger_metrics.stage_percentiles(s);
                let q = stage(LatencyStage::Queue);
                let t = stage(LatencyStage::Tokenize);
                let f = stage(LatencyStage::Forward);
                let tot = stage(LatencyStage::Total);
                eprintln!(
                    "llm-d-sc metrics: served={served} hits={} misses={} coalesced={} | \
                     queue p50={:?} p99={:?} | tokenize p50={:?} p99={:?} | \
                     forward p50={:?} p99={:?} | total p50={:?} p99={:?}",
                    snap.cache_hits,
                    snap.cache_misses,
                    snap.cache_coalesced,
                    q.p50,
                    q.p99,
                    t.p50,
                    t.p99,
                    f.p50,
                    f.p99,
                    tot.p50,
                    tot.p99
                );
            }
        })
        .expect("metrics log thread must spawn");

    // Block until SIGTERM (Kubernetes pod termination) or SIGINT (Ctrl-C).
    // A second signal during the drain falls back to the default disposition
    // (immediate termination) because this watcher runtime is gone by then.
    let signal = wait_for_termination_signal();
    let grace = std::time::Duration::from_secs(
        env::var(SHUTDOWN_GRACE_ENV)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_SHUTDOWN_GRACE_SECS),
    );
    eprintln!("llm-d-sc: {signal} received; readiness -> DRAINING, grace {grace:?}");

    // Stop the periodic logger (its recv_timeout wakes on disconnect), then
    // drain: admission stops, in-flight classifications finish, connections
    // close — bounded by the grace period.
    drop(stop_tx);
    metrics_logger
        .join()
        .expect("metrics logger thread must not panic");
    let drain = server.shutdown_blocking(grace);

    // Final metrics line AFTER the drain, so the totals include everything the
    // drain completed (the evidence a shutdown log line can carry).
    let snap = metrics.snapshot();
    let served = snap.cache_hits + snap.cache_misses + snap.cache_coalesced;
    let stage = |s| metrics.stage_percentiles(s);
    let q = stage(LatencyStage::Queue);
    let t = stage(LatencyStage::Tokenize);
    let f = stage(LatencyStage::Forward);
    let tot = stage(LatencyStage::Total);
    eprintln!(
        "llm-d-sc final metrics: served={served} hits={} misses={} coalesced={} | \
         queue p50={:?} p99={:?} | tokenize p50={:?} p99={:?} | \
         forward p50={:?} p99={:?} | total p50={:?} p99={:?}",
        snap.cache_hits,
        snap.cache_misses,
        snap.cache_coalesced,
        q.p50,
        q.p99,
        t.p50,
        t.p99,
        f.p50,
        f.p99,
        tot.p50,
        tot.p99
    );

    match drain {
        Ok(()) => {
            eprintln!("llm-d-sc: drained in-flight work; exiting");
        }
        // The grace period elapsed. Whatever was still open has been severed;
        // exiting 0 is still correct (the process DID stop, within its bound)
        // but the log says plainly that work may have been cut off.
        Err(e) => {
            eprintln!("llm-d-sc: {e}");
        }
    }
    Ok(())
}

/// Block until the process is asked to terminate.
///
/// SIGTERM is what Kubernetes delivers on pod termination, so it is the signal
/// graceful shutdown exists for; SIGINT makes interactive Ctrl-C behave the
/// same. Implemented on a throwaway single-threaded Tokio runtime because
/// signal delivery is async I/O; the serving runtime belongs to the
/// [`ClassifyServer`].
fn wait_for_termination_signal() -> &'static str {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("signal watcher runtime must build");
    runtime.block_on(async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut terminate =
                signal(SignalKind::terminate()).expect("SIGTERM handler must install");
            let mut interrupt =
                signal(SignalKind::interrupt()).expect("SIGINT handler must install");
            tokio::select! {
                _ = terminate.recv() => "SIGTERM",
                _ = interrupt.recv() => "SIGINT",
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c()
                .await
                .expect("Ctrl-C handler must install");
            "Ctrl-C"
        }
    })
}
