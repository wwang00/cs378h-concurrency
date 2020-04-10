//!
//! participant.rs
//! Implementation of 2PC participant
//!
extern crate log;
extern crate rand;
extern crate stderrlog;
use message::MessageType;
use message::ProtocolMessage;
use message::RequestStatus;
use oplog;
use participant::rand::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

///
/// ParticipantState
/// enum for participant 2PC state machine
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticipantState {
    Quiescent,
    Armed,
    Ready,
    Abort,
    Commit,
}

///
/// Participant
/// structure for maintaining per-participant state
/// and communication/synchronization objects to/from coordinator
///
#[derive(Debug)]
pub struct Participant {
    id: i32,
    id_string: String,
    state: ParticipantState,
    running: Arc<AtomicBool>,
    log: oplog::OpLog,
    success_prob_ops: f64,
    success_prob_msg: f64,
    tx: Sender<ProtocolMessage>,
    rx: Receiver<ProtocolMessage>,
    committed: i32,
    aborted: i32,
    unknown: i32,
}

// static timeout for receiving result from coordinator
static TIMEOUT: Duration = Duration::from_millis(500);

///
/// Participant
/// implementation of per-participant 2PC protocol
/// Required:
/// 1. new -- ctor
/// 2. pub fn report_status -- reports number of committed/aborted/unknown for each participant
/// 3. pub fn protocol() -- implements participant side protocol
///
impl Participant {
    ///
    /// new()
    ///
    /// Return a new participant, ready to run the 2PC protocol
    /// with the coordinator.
    ///
    /// HINT: you may want to pass some channels or other communication
    ///       objects that enable coordinator->participant and participant->coordinator
    ///       messaging to this ctor.
    /// HINT: you may want to pass some global flags that indicate whether
    ///       the protocol is still running to this constructor. There are other
    ///       ways to communicate this, of course.
    ///
    pub fn new(
        id: i32,
        id_string: String,
        running: Arc<AtomicBool>,
        logpath: String,
        success_prob_ops: f64,
        success_prob_msg: f64,
        tx: Sender<ProtocolMessage>,
        rx: Receiver<ProtocolMessage>,
    ) -> Participant {
        Participant {
            id,
            id_string,
            state: ParticipantState::Quiescent,
            running,
            log: oplog::OpLog::new(logpath),
            success_prob_ops,
            success_prob_msg,
            tx,
            rx,
            committed: 0,
            aborted: 0,
            unknown: 0,
        }
    }

    ///
    /// send()
    /// Send a protocol message to the coordinator.
    /// This variant can be assumed to always succeed.
    /// You should make sure your solution works using this
    /// variant before working with the send_unreliable variant.
    ///
    /// HINT: you will need to implement something that does the
    ///       actual sending.
    ///
    pub fn send(&self, pm: ProtocolMessage) -> bool {
        trace!("participant_{}::send...", self.id);
        let result: bool = true;
        self.tx.send(pm).unwrap();
        trace!("participant_{}::send exit", self.id);
        result
    }

    ///
    /// send()
    /// Send a protocol message to the coordinator,
    /// with some probability of success thresholded by the
    /// command line option success_probability [0.0..1.0].
    /// This variant can be assumed to always succeed
    ///
    /// HINT: you will need to implement something that does the
    ///       actual sending, but you can use the threshold
    ///       logic in this implementation below.
    ///
    pub fn send_unreliable(&self, pm: ProtocolMessage) -> bool {
        let x: f64 = random();
        let result: bool;
        if x < self.success_prob_msg {
            result = self.send(pm);
        } else {
            result = false;
        }
        result
    }

    ///
    /// recv_request()
    /// receive a message from coordinator
    ///
    pub fn recv_request(&self) -> Option<ProtocolMessage> {
        trace!("participant_{}::recv_request...", self.id);
        let mut result = Option::None;
        assert!(self.state == ParticipantState::Quiescent);

        let received = self.rx.try_recv();
        if let Ok(pm) = received {
            info!("participant_{}  received {:?}", self.id, pm);
            result = Some(pm);
        }
        trace!("participant_{}::recv_request exit", self.id);
        result
    }

    ///
    /// perform_operation
    /// perform the operation specified in the 2PC proposal,
    /// with some probability of success/failure determined by the
    /// command-line option success_probability.
    ///
    /// HINT: The code provided here is not complete--it provides some
    ///       tracing infrastructure and the probability logic.
    ///       Your implementation need not preserve the method signature
    ///       (it's ok to add parameters or return something other than
    ///       bool if it's more convenient for your design).
    ///
    pub fn perform_operation(&self) -> bool {
        trace!("participant_{}::perform_operation", self.id);
        assert!(self.state == ParticipantState::Armed);

        let mut result: RequestStatus = RequestStatus::Unknown;

        let x: f64 = random();
        if x < self.success_prob_ops {
            info!("participant_{}  success", self.id);
            result = RequestStatus::Committed;
        } else {
            info!("participant_{}  failure", self.id);
        }

        trace!("participant_{}::perform_operation exit", self.id);
        result == RequestStatus::Committed
    }

    ///
    /// report_status()
    /// report the abort/commit/unknown status (aggregate) of all
    /// transaction requests made by this coordinator before exiting.
    ///
    pub fn report_status(&self) {
        println!(
            "participant_{}:\tC:{}\tA:{}\tU:{}",
            self.id, self.committed, self.aborted, self.unknown
        );
    }

    ///
    /// protocol()
    /// Implements the participant side of the 2PC protocol
    /// HINT: if the simulation ends early, don't keep handling requests!
    /// HINT: wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self) {
        trace!("participant_{}::protocol", self.id);

        while self.running.load(Ordering::SeqCst) {
            // get request
            let request = self.recv_request();
            if let None = request {
                continue;
            }
            let request = request.unwrap();
            let txid = request.txid;
            let senderid = request.senderid.clone();
            let opid = request.opid;
            self.state = ParticipantState::Armed;

            // perform operation
            let op_success = self.perform_operation();
            self.state = ParticipantState::Ready;

            // vote
            let vote: ProtocolMessage;
            let mtype: MessageType;
            let state: ParticipantState;
            if op_success {
                mtype = MessageType::ParticipantVoteCommit;
                state = ParticipantState::Commit;
            } else {
                mtype = MessageType::ParticipantVoteAbort;
                state = ParticipantState::Abort;
            }
            self.log.append(mtype, txid, senderid.clone(), opid);
            vote = ProtocolMessage::generate(mtype, txid, senderid.clone(), opid);
            self.send(vote);
            self.state = state;

            // get final decision, update locally
            let decision = self.rx.recv().unwrap();
            match decision.mtype {
                MessageType::CoordinatorCommit => self.committed += 1,
                MessageType::CoordinatorAbort => self.aborted += 1,
                _ => (),
            }
            self.log
                .append(decision.mtype, txid, senderid.clone(), opid);
            self.state = ParticipantState::Quiescent;
        }

        self.report_status();

        trace!("participant_{}  exiting", self.id);
    }
}
