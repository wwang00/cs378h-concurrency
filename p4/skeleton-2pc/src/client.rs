//!
//! client.rs
//! Implementation of 2PC client
//!
extern crate log;
extern crate stderrlog;
use message::MessageType;
use message::ProtocolMessage;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

// static counter for getting unique TXID numbers
static TXID_COUNTER: AtomicI32 = AtomicI32::new(1);

// client state and
// primitives for communicating with
// the coordinator
pub struct Client {
    id: i32,
    id_string: String,
    running: Arc<AtomicBool>,
    n_clients: i32,
    tx: Sender<ProtocolMessage>,
    rx: Receiver<ProtocolMessage>,
    committed: i32,
    aborted: i32,
    unknown: i32,
}

///
/// client implementation
/// Required:
/// 1. new -- ctor
/// 2. pub fn report_status -- reports number of committed/aborted/unknown
/// 3. pub fn protocol(&mut self, n_requests: i32) -- implements client side protocol
///
impl Client {
    ///
    /// new()
    ///
    /// Return a new client, ready to run the 2PC protocol
    /// with the coordinator.
    ///
    /// HINT: you may want to pass some channels or other communication
    ///       objects that enable coordinator->client and client->coordinator
    ///       messaging to this ctor.
    /// HINT: you may want to pass some global flags that indicate whether
    ///       the protocol is still running to this constructor
    ///
    pub fn new(
        id: i32,
        id_string: String,
        running: Arc<AtomicBool>,
        n_clients: i32,
        tx: Sender<ProtocolMessage>,
        rx: Receiver<ProtocolMessage>,
    ) -> Client {
        Client {
            id,
            id_string,
            running,
            n_clients,
            tx,
            rx,
            committed: 0,
            aborted: 0,
            unknown: 0,
        }
    }

    ///
    /// send_next_operation(&mut self)
    /// send the next operation to the coordinator
    ///
    pub fn send_next_operation(&self, opid: i32) -> Option<()> {
        trace!("{}::send_next_operation...", self.id_string);

        // create a new request with a unique TXID.
        let txid = TXID_COUNTER.fetch_add(1, Ordering::SeqCst);

        info!(
            "{}  request({})->txid:{} called",
            self.id_string, opid, txid
        );
        let pm = ProtocolMessage::generate(
            MessageType::ClientRequest,
            txid,
            format!("{}", self.id_string),
            opid,
        );

        info!("\t{}  calling send...", self.id_string);

        let result = self.tx.send(pm);

        trace!("{}::send_next_operation exit", self.id_string);
        if let Ok(()) = result {
            return Some(());
        }
        None
    }

    ///
    /// recv_result()
    /// Wait for the coordinator to respond with the result for the
    /// last issued request. Note that we assume the coordinator does
    /// not fail in this simulation
    ///
    pub fn recv_result(&self) -> Option<ProtocolMessage> {
        trace!("{}::recv_result...", self.id_string);
        let mut result = Option::None;

        let received = self.rx.recv();
        if let Ok(pm) = received {
            info!("{}  received {:?}", self.id_string, pm);
            result = Some(pm);
        }
        trace!("{}::recv_result exit", self.id_string);
        result
    }

    ///
    /// wait_for_exit_signal(&mut self)
    /// wait until the running flag is set by the CTRL-C handler
    ///
    pub fn wait_for_exit_signal(&self) {
        trace!("{}::wait_for_exit_signal", self.id_string);

        while self.running.load(Ordering::SeqCst) {}
    }

    ///
    /// report_status()
    /// report the abort/commit/unknown status (aggregate) of all
    /// transaction requests made by this client before exiting.
    ///
    pub fn report_status(&self) {
        println!(
            "{}:\tC:{}\tA:{}\tU:{}",
            self.id_string, self.committed, self.aborted, self.unknown
        );
    }

    ///
    /// protocol()
    /// Implements the client side of the 2PC protocol
    /// HINT: if the simulation ends early, don't keep issuing requests!
    /// HINT: if you've issued all your requests, wait for some kind of
    ///       exit signal before returning from the protocol method!
    ///
    pub fn protocol(&mut self, n_requests: i32) {
        trace!("{}::protocol", self.id_string);

        // send all requests
        for r in 0..n_requests {
            let opid = r * self.n_clients + self.id;
            let result = self.send_next_operation(opid);
            if let None = result {
                panic!("COORDINATOR DISCONNECTED");
            }
        }

        // receive results
        loop {
            let result = self.recv_result();
            if let None = result {
                panic!("COORDINATOR DISCONNECTED");
            }
            let result = result.unwrap();
            info!("{}  result {:?}", self.id_string, result);
            match result.mtype {
                MessageType::CoordinatorCommit => self.committed += 1,
                MessageType::CoordinatorAbort => self.aborted += 1,
                MessageType::CoordinatorExit => break,
                _ => (),
            }
        }

        self.wait_for_exit_signal();
        self.report_status();

        trace!("{}  exiting", self.id_string);
    }
}
