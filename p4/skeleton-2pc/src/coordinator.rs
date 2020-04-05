//!
//! coordinator.rs
//! Implementation of 2PC coordinator
//!
extern crate log;
extern crate rand;
extern crate stderrlog;
use coordinator::rand::prelude::*;
use message;
use message::MessageType;
use message::ProtocolMessage;
use message::RequestStatus;
use oplog;
use std::collections::HashMap;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::mpsc::channel;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

/// CoordinatorState
/// States for 2PC state machine
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinatorState {
    Quiescent,
    Wait,
    Abort,
    Commit,
}

/// Coordinator
/// struct maintaining state for coordinator
#[derive(Debug)]
pub struct Coordinator {
    state: CoordinatorState,
    running: Arc<AtomicBool>,
    log: oplog::OpLog,
    success_prob_ops: f64,
    success_prob_msg: f64,
    n_clients: i32,
    n_requests: i32,
    n_participants: i32,
    tx_clients: Vec<Sender<ProtocolMessage>>,
    rx_clients: Vec<Receiver<ProtocolMessage>>,
    tx_participants: Vec<Sender<ProtocolMessage>>,
    rx_participants: Vec<Receiver<ProtocolMessage>>,
}

///
/// Coordinator
/// implementation of coordinator functionality
/// Required:
/// 1. new -- ctor
/// 2. protocol -- implementation of coordinator side of protocol
/// 3. report_status -- report of aggregate commit/abort/unknown stats on exit.
/// 4. participant_join -- what to do when a participant joins
/// 5. client_join -- what to do when a client joins
///
impl Coordinator {
    ///
    /// new()
    /// Initialize a new coordinator
    ///
    pub fn new(
        running: Arc<AtomicBool>,
        logpath: String,
        success_prob_ops: f64,
        success_prob_msg: f64,
        n_clients: i32,
        n_requests: i32,
        n_participants: i32,
    ) -> Coordinator {
        Coordinator {
            state: CoordinatorState::Quiescent,
            running,
            log: oplog::OpLog::new(logpath),
            success_prob_ops,
            success_prob_msg,
            n_clients,
            n_requests,
            n_participants,
            tx_clients: Vec::new(),
            rx_clients: Vec::new(),
            tx_participants: Vec::new(),
            rx_participants: Vec::new(),
        }
    }

    ///
    /// participant_join()
    /// handle the addition of a new participant
    /// HINT: keep track of any channels involved!
    /// HINT: you'll probably need to change this routine's
    ///       signature to return something!
    ///       (e.g. channel(s) to be used)
    ///
    pub fn participant_join(&mut self) -> (Sender<ProtocolMessage>, Receiver<ProtocolMessage>) {
        assert!(self.state == CoordinatorState::Quiescent);
        let (tx_coordinator, rx_participant) = channel();
        let (tx_participant, rx_coordinator) = channel();
        self.tx_participants.push(tx_coordinator);
        self.rx_participants.push(rx_coordinator);

        (tx_participant, rx_participant)
    }

    ///
    /// client_join()
    /// handle the addition of a new client
    /// HINTS: keep track of any channels involved!
    /// HINT: you'll probably need to change this routine's
    ///       signature to return something!
    ///       (e.g. channel(s) to be used)
    ///
    pub fn client_join(&mut self) -> (Sender<ProtocolMessage>, Receiver<ProtocolMessage>) {
        assert!(self.state == CoordinatorState::Quiescent);
        let (tx_coordinator, rx_client) = channel();
        let (tx_client, rx_coordinator) = channel();
        self.tx_participants.push(tx_coordinator);
        self.rx_participants.push(rx_coordinator);

        (tx_client, rx_client)
    }

    ///
    /// send()
    /// send a message, maybe drop it
    /// HINT: you'll need to do something to implement
    ///       the actual sending!
    ///
    pub fn send(&mut self, sender: &Sender<ProtocolMessage>, pm: ProtocolMessage) -> bool {
        let x: f64 = random();
        let mut result: bool = true;
        if x < self.success_prob_ops {
            sender.send(pm).unwrap();
        } else {
            // don't send anything!
            // (simulates failure)
            result = false;
        }
        result
    }

    ///
    /// recv_request()
    /// receive a message from a client
    /// to start off the protocol.
    ///
    pub fn recv_request(&mut self) -> Option<ProtocolMessage> {
        let mut result = Option::None;
        assert!(self.state == CoordinatorState::Quiescent);
        trace!("coordinator::recv_request...");

        // TODO: write me!

        trace!("leaving coordinator::recv_request");
        result
    }

    ///
    /// report_status()
    /// report the abort/commit/unknown status (aggregate) of all
    /// transaction requests made by this coordinator before exiting.
    ///
    pub fn report_status(&mut self) {
        let successful_ops: usize = 0; // TODO!
        let failed_ops: usize = 0; // TODO!
        let unknown_ops: usize = 0; // TODO!
        println!(
            "coordinator:\tC:{}\tA:{}\tU:{}",
            successful_ops, failed_ops, unknown_ops
        );
    }

    ///
    /// protocol()
    /// Implements the coordinator side of the 2PC protocol
    /// HINT: if the simulation ends early, don't keep handling requests!
    /// HINT: wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self) {
        while self.running.load(Ordering::SeqCst) {}

        self.report_status();
    }
}
