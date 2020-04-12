//!
//! participant.rs
//! Implementation of 2PC participant
//!
extern crate log;
extern crate rand;
extern crate stderrlog;
use message::MessageType;
use message::ProtocolMessage;
use oplog;
use participant::rand::prelude::*;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

///
/// ParticipantState
/// enum for participant 2PC state machine
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticipantState {
    Quiescent,
    Ready,
    Decided,
}

///
/// Participant
/// structure for maintaining per-participant state
/// and communication/synchronization objects to/from coordinator
///
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
    ///
    pub fn send(&self, pm: ProtocolMessage) -> bool {
        trace!("participant_{}::send...", self.id);
        let result: bool = true;
        self.tx.send(pm).unwrap();
        trace!("participant_{}::send exit", self.id);
        result
    }

    ///
    /// send_unreliable()
    /// Send a protocol message to the coordinator,
    /// with some probability of success thresholded by the
    /// command line option success_probability [0.0..1.0].
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
        assert!(self.state != ParticipantState::Ready);

        let mut result = Option::None;
        let received = self.rx.recv();
        if let Ok(pm) = received {
            info!(
                "participant_{} ({:?})  received {:?}",
                self.id, self.state, pm
            );
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
    pub fn perform_operation(&self) -> bool {
        trace!("participant_{}::perform_operation", self.id);
        assert!(self.state == ParticipantState::Ready);

        let mut result = false;

        let x: f64 = random();
        if x < self.success_prob_ops {
            info!("participant_{}  success", self.id);
            result = true;
        } else {
            info!("participant_{}  failure", self.id);
        }

        trace!("participant_{}::perform_operation exit", self.id);
        result
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

        let mut txid = -1;
        let mut senderid = String::new();
        let mut opid = -1;

        loop {
            match self.state {
                ParticipantState::Quiescent => {
                    // get request
                    let request = self.recv_request();
                    if let None = request {
                        // TODO handle coordinator failure
                        panic!("COORDINATOR FAILED");
                        // break;
                    }
                    let request = request.unwrap();
                    match request.mtype {
                        MessageType::CoordinatorPropose => (),
                        MessageType::CoordinatorExit => break,
                        MessageType::CoordinatorCommit => {
                            panic!("participant received commit in quiescent")
                        }
                        _ => continue,
                    }
                    txid = request.txid;
                    senderid = request.senderid;
                    opid = request.opid;
                    self.state = ParticipantState::Ready;
                }
                ParticipantState::Ready => {
                    // perform operation
                    let op_success = self.perform_operation();

                    // vote
                    let vote: ProtocolMessage;
                    let mtype = if op_success {
                        MessageType::ParticipantVoteCommit
                    } else {
                        MessageType::ParticipantVoteAbort
                    };
                    self.log.append(mtype, txid, senderid.clone(), opid);
                    vote = ProtocolMessage::generate(mtype, txid, senderid.clone(), opid);
                    self.send_unreliable(vote);
                    self.state = ParticipantState::Decided;
                }
                ParticipantState::Decided => {
                    // get final decision, update locally
                    let decision = self.recv_request();
                    if let None = decision {
                        // TODO handle coordinator failure
                        panic!("COORDINATOR FAILED");
                        // break;
                    }
                    let decision = decision.unwrap();
                    match decision.mtype {
                        MessageType::CoordinatorCommit => self.committed += 1,
                        MessageType::CoordinatorAbort => self.aborted += 1,
                        MessageType::CoordinatorExit => break,
                        _ => (),
                    }
                    self.log
                        .append(decision.mtype, txid, senderid.clone(), opid);
                    self.state = ParticipantState::Quiescent;
                }
            }
        }

        self.report_status();

        trace!("participant_{}  exiting", self.id);
    }
}
