//!
//! client.rs
//! Implementation of 2PC client
//!
extern crate log;
extern crate stderrlog;
use message;
use message::MessageType;
use message::ProtocolMessage;
use message::RequestStatus;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// static counter for getting unique TXID numbers
static TXID_COUNTER: AtomicI32 = AtomicI32::new(1);

// client state and
// primitives for communicating with
// the coordinator
#[derive(Debug)]
pub struct Client {
    id: i32,
    id_string: String,
    running: Arc<AtomicBool>,
    tx: Sender<ProtocolMessage>,
    rx: Receiver<ProtocolMessage>,
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
        tx: Sender<ProtocolMessage>,
        rx: Receiver<ProtocolMessage>,
    ) -> Client {
        Client {
            id,
            id_string,
            running,
            tx,
            rx,
        }
    }

    ///
    /// wait_for_exit_signal(&mut self)
    /// wait until the running flag is set by the CTRL-C handler
    ///
    pub fn wait_for_exit_signal(&mut self) {
        trace!("client_{} waiting for exit signal", self.id);

        while self.running.load(Ordering::SeqCst) {}

        trace!("client_{} exiting", self.id);
    }

    ///
    /// send_next_operation(&mut self)
    /// send the next operation to the coordinator
    ///
    pub fn send_next_operation(&mut self) {
        trace!("client_{}::send_next_operation", self.id);

        // create a new request with a unique TXID.
        let request_no: i32 = 0; // TODO--choose another number!
        let txid = TXID_COUNTER.fetch_add(1, Ordering::SeqCst);

        info!(
            "Client {} request({})->txid:{} called",
            self.id, request_no, txid
        );
        let pm = message::ProtocolMessage::generate(
            message::MessageType::ClientRequest,
            txid,
            format!("client_{}", self.id),
            request_no,
        );

        info!("client {} calling send...", self.id);

        // TODO

        trace!("client_{}::exit send_next_operation", self.id);
    }

    ///
    /// recv_result()
    /// Wait for the coordinator to respond with the result for the
    /// last issued request. Note that we assume the coordinator does
    /// not fail in this simulation
    ///
    pub fn recv_result(&mut self) {
        trace!("client_{}::recv_result", self.id);

        // TODO

        trace!("client_{}::exit recv_result", self.id);
    }

    ///
    /// report_status()
    /// report the abort/commit/unknown status (aggregate) of all
    /// transaction requests made by this client before exiting.
    ///
    pub fn report_status(&mut self) {
        // TODO: collect real stats!
        let successful_ops: usize = 0;
        let failed_ops: usize = 0;
        let unknown_ops: usize = 0;
        println!(
            "client_{}:\tC:{}\tA:{}\tU:{}",
            self.id, successful_ops, failed_ops, unknown_ops
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
        // run the 2PC protocol for each of n_requests

        // TODO

        // wait for signal to exit
        // and then report status
        self.wait_for_exit_signal();
        self.report_status();
    }
}
