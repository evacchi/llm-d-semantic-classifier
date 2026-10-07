//! Graceful shutdown (drain) proving tests.
//!
//! U-035 (0.20 runtime-hardening, delivered with the graceful-shutdown work):
//! shutdown stops admission and drains configured in-flight work. The drain is
//! observed from OUTSIDE the shutdown code: a request whose forward is held at
//! a gate must COMPLETE (not be dropped) after shutdown begins, while a NEW
//! connection attempted after shutdown must never be served.
//!
//! R-015: repeated start/shutdown sequences do not deadlock. A restart loop
//! that binds, serves, and drains in place would hang if shutdown leaked a
//! task, permit, or thread, so the loop itself is the assertion.
//!
//! The readiness flip (I-013) is asserted inside U-035 as a precondition: the
//! server must report DRAINING while in-flight work is still running, i.e.
//! BEFORE the drain completes.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use llm_d_sc::classify::{
    ClassificationInput, ClassificationResult, ClassifierRuntime, ClassifyError, ClassifyStatus,
    Embedding, RankedSignal, RuntimeMetadata,
};
use llm_d_sc::grpc::classify::generated;
use llm_d_sc::grpc::classify::{ClassifyClient, ClassifyRequest, ClassifyResponse, ClassifyServer};
use llm_d_sc::runtime::Readiness;

/// A one-shot gate: `wait_open` blocks until `open` is called.
///
/// The classifier's forward blocks on this so a request can be held IN FLIGHT
/// deterministically (a sleep would race the shutdown path).
#[derive(Default)]
struct Gate {
    opened: Mutex<bool>,
    cv: Condvar,
}

impl Gate {
    fn open(&self) {
        let mut opened = self.opened.lock().expect("gate lock");
        *opened = true;
        self.cv.notify_all();
    }

    fn wait_open(&self) {
        let mut opened = self.opened.lock().expect("gate lock");
        while !*opened {
            opened = self.cv.wait(opened).expect("gate condvar");
        }
    }
}

/// A classifier whose forward (the embed half) blocks on a gate and counts
/// entries/exits.
///
/// `ServiceCore` drives cache misses through `embed` then `rank`, so the gate
/// lives in `embed` (the model forward) and `rank` returns the canned result:
/// entered increments when the forward starts, completed when it finishes.
struct GatedClassifier {
    gate: Arc<Gate>,
    /// Number of forwards that ENTERED (admitted work reaching the model).
    entered: Arc<AtomicUsize>,
    /// Number of forwards that COMPLETED (drained work, not dropped).
    completed: Arc<AtomicUsize>,
}

impl ClassifierRuntime for GatedClassifier {
    fn metadata(&self) -> RuntimeMetadata {
        RuntimeMetadata {
            classifier_id: "test-gated".into(),
            signal: "sensitivity".into(),
            model_revision: "test".into(),
            tokenizer_revision: "test".into(),
            taxonomy_revision: "test".into(),
            artifact_digest: None,
            ranking_mode: llm_d_sc::classify::RankingMode::AnchorCosine,
        }
    }

    fn embed(&self, _input: &ClassificationInput) -> Result<Embedding, ClassifyError> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        self.gate.wait_open();
        Ok(Embedding {
            vector: vec![1.0, 0.0],
            logits: None,
        })
    }

    fn rank(
        &self,
        _embedding: &Embedding,
        _input: &ClassificationInput,
    ) -> Result<ClassificationResult, ClassifyError> {
        self.completed.fetch_add(1, Ordering::SeqCst);
        Ok(ClassificationResult {
            classifier_id: "test-gated".into(),
            model_revision: "test".into(),
            tokenizer_revision: "test".into(),
            taxonomy_revision: "test".into(),
            status: ClassifyStatus::Ok,
            ranked: vec![RankedSignal {
                id: "high".into(),
                score: 0.99,
            }],
        })
    }

    fn classify(&self, _input: ClassificationInput) -> Result<ClassificationResult, ClassifyError> {
        // Never reached: ServiceCore owns the cache/single-flight path and
        // calls embed + rank directly. Panicking here would fail the test
        // loudly rather than letting a bypass go unnoticed.
        unimplemented!("ServiceCore must drive embed + rank, not classify")
    }
}

fn fixture_request(request_id: &str, context: &str) -> ClassifyRequest {
    ClassifyRequest {
        request_id: request_id.to_string(),
        session_id: "sess-drain".to_string(),
        context: context.to_string(),
        signals: Vec::new(),
        context_completeness: generated::ContextCompleteness::Full as i32,
    }
}

/// Poll until `f` holds, for at most `budget`. Returns false on timeout.
fn soon(budget: Duration, mut f: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    f()
}

/// U-035: shutdown stops admission and drains configured in-flight work.
///
/// One request is held IN FLIGHT inside the model forward when shutdown
/// begins. The contract under test, observed from outside:
/// - readiness flips to DRAINING while the request is still running (I-013);
/// - a NEW connection attempted after shutdown is never admitted (no second
///   forward ever starts, and the late client's call fails);
/// - the in-flight request COMPLETES with its ranked signals (drained, not
///   dropped), running exactly one forward;
/// - the bounded drain returns cleanly within its grace.
#[test]
fn u035_shutdown_stops_admission_and_drains_in_flight() {
    let gate = Arc::new(Gate::default());
    let entered = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicUsize::new(0));
    let server = ClassifyServer::bind_with_runtime(
        "127.0.0.1:0",
        GatedClassifier {
            gate: gate.clone(),
            entered: entered.clone(),
            completed: completed.clone(),
        },
    )
    .expect("server must bind with the gated runtime");
    let addr = server.local_addr();

    // Client A: one request whose forward blocks on the gate.
    let mut client_a =
        ClassifyClient::connect(&addr).expect("client A must connect before shutdown");
    let inflight = std::thread::spawn(move || {
        client_a.classify(fixture_request(
            "req-inflight",
            "held-open in-flight classification a",
        ))
    });

    // Establish the in-flight forward before shutting down.
    assert!(
        soon(Duration::from_secs(5), || entered.load(Ordering::SeqCst)
            == 1),
        "the request must reach the model forward before shutdown begins"
    );

    // Begin graceful shutdown. Admission stops; the in-flight forward is STILL
    // blocked on the gate at this point.
    server.shutdown();

    // I-013 precondition: the readiness flip is observable BEFORE the drain
    // completes (the forward has not finished — it is still held at the gate).
    assert_eq!(server.readiness(), Readiness::Draining);
    assert_eq!(completed.load(Ordering::SeqCst), 0);

    // Give the serve task a moment to leave the accept loop, then attempt a
    // NEW connection. It must never be admitted: no second forward may start.
    // (Generous for a loaded CI runner: the watch wake schedules the serve
    // task immediately, but this must not depend on that being instant.)
    std::thread::sleep(Duration::from_millis(200));
    let late_addr = addr.clone();
    let late = std::thread::spawn(move || {
        let mut client_b = ClassifyClient::connect(&late_addr)?;
        client_b.classify(fixture_request("req-late", "late arriving request b"))
    });
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(
        entered.load(Ordering::SeqCst),
        1,
        "admission must stop at shutdown: the late request must never start a forward"
    );

    // Release the gate: the in-flight request must COMPLETE (drained, not
    // dropped) with its ranked signals.
    gate.open();
    let response = inflight
        .join()
        .expect("in-flight request thread must not panic")
        .expect("the in-flight request must complete during the drain, not be dropped");
    assert_eq!(
        response.status,
        generated::ClassificationStatus::Ok as i32,
        "the drained request must succeed"
    );
    assert_eq!(
        response.ranked.len(),
        1,
        "the drained response must carry its ranked signals"
    );

    // Exactly ONE forward ran and it COMPLETED: in-flight work drained, late
    // work was never admitted. Snapshotted before the consuming drain call;
    // the drained request is already counted (it completed above).
    let snap = server.metrics_snapshot();

    // The bounded drain returns cleanly (all connections closed) within grace.
    let drained = server.shutdown_blocking(Duration::from_secs(10));
    assert!(
        drained.is_ok(),
        "the drain must complete cleanly once the in-flight request finished: {drained:?}"
    );

    // The late client's call fails (it was never admitted; once the drain
    // finished its connection was closed under it).
    let late_result = late.join().expect("late client thread must not panic");
    assert!(
        late_result.is_err(),
        "a request arriving after shutdown must not be served"
    );

    // Exactly ONE forward ran and it COMPLETED: in-flight work drained, late
    // work was never admitted.
    assert_eq!(entered.load(Ordering::SeqCst), 1);
    assert_eq!(completed.load(Ordering::SeqCst), 1);
    assert_eq!(
        snap.cache_misses, 1,
        "exactly one forward must have run (the drained in-flight request)"
    );
}

/// U-035 (admission-stop half, stated positively): while the server is up, a
/// second request over a SECOND connection IS admitted. This is the control
/// for the late-client assertion above — it fails if the late request is
/// rejected for any reason other than shutdown.
#[test]
fn u035_control_second_connection_is_served_before_shutdown() {
    let gate = Arc::new(Gate::default());
    let entered = Arc::new(AtomicUsize::new(0));
    let server = ClassifyServer::bind_with_runtime(
        "127.0.0.1:0",
        GatedClassifier {
            gate: gate.clone(),
            entered: entered.clone(),
            completed: Arc::new(AtomicUsize::new(0)),
        },
    )
    .expect("server must bind with the gated runtime");
    let addr = server.local_addr();

    // Client A blocks in the forward; client B is admitted on a second
    // connection and ALSO reaches the forward (worker width > 1 by default).
    let mut client_a = ClassifyClient::connect(&addr).expect("client A must connect");
    // Distinct contexts per request: the exact-result cache and single-flight
    // coalescing must not be able to hide a second admission behind the first.
    let a = std::thread::spawn(move || client_a.classify(fixture_request("req-a", "control a")));
    assert!(
        soon(Duration::from_secs(5), || entered.load(Ordering::SeqCst)
            == 1),
        "client A's forward must start"
    );
    let mut client_b = ClassifyClient::connect(&addr).expect("client B must connect");
    let b = std::thread::spawn(move || client_b.classify(fixture_request("req-b", "control b")));
    assert!(
        soon(Duration::from_secs(5), || entered.load(Ordering::SeqCst)
            == 2),
        "without shutdown, a second connection's request must be admitted"
    );

    // Both complete and drain cleanly.
    gate.open();
    let response: ClassifyResponse = a
        .join()
        .expect("client A thread must not panic")
        .expect("client A's request must succeed");
    assert_eq!(response.status, generated::ClassificationStatus::Ok as i32);
    let response: ClassifyResponse = b
        .join()
        .expect("client B thread must not panic")
        .expect("client B's request must succeed");
    assert_eq!(response.status, generated::ClassificationStatus::Ok as i32);
    assert!(
        server.shutdown_blocking(Duration::from_secs(10)).is_ok(),
        "the drain must complete cleanly"
    );
}

/// R-015: repeated start/shutdown sequences do not deadlock.
///
/// Five bind -> serve -> drain cycles with a client connection still open at
/// drain time. A leaked task, watch receiver, permit, or executor thread would
/// either hang a drain past its grace or make a later bind/serve cycle stall;
/// the loop finishing quickly inside the budget is the assertion.
#[test]
fn r015_repeated_start_shutdown_does_not_deadlock() {
    const CYCLES: usize = 5;
    let started = Instant::now();
    for cycle in 0..CYCLES {
        let server = ClassifyServer::bind("127.0.0.1:0").expect("server must bind");
        let addr = server.local_addr();
        let mut client = ClassifyClient::connect(&addr).expect("client must connect");

        let response = client
            .classify(fixture_request(
                &format!("req-cycle-{cycle}"),
                &format!("cycle {cycle} request"),
            ))
            .expect("classify must succeed before shutdown");
        assert_eq!(response.status, generated::ClassificationStatus::Ok as i32);

        // Drain while the client's persistent channel is still OPEN — the
        // hard case: the server must close its side, not wait for the client.
        let cycle_started = Instant::now();
        server
            .shutdown_blocking(Duration::from_secs(10))
            .expect("each drain must complete within its grace");
        assert!(
            cycle_started.elapsed() < Duration::from_secs(2),
            "drain of an idle open connection must be fast, not wait out the grace"
        );

        // The connection is dead after the drain: the client must not be able
        // to keep using it (admission really stopped).
        assert!(
            client
                .classify(fixture_request("req-after-drain", "post-drain request"))
                .is_err(),
            "a post-drain request over the old channel must fail"
        );
    }
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "the full start/shutdown cycle loop must finish well inside the budget"
    );
}
