//!
//! participant.rs
//! Implementation of 2PC participant
//!
extern crate log;
extern crate rand;
extern crate stderrlog;
use message::MessageType;
use message::ProtocolMessage;
use oplog::OpLog;
use participant::rand::prelude::*;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

///
/// ParticipantState
/// enum for participant 2PC state machine
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticipantState {
    Quiescent,
    Voting,
    Confirming,
    Recovering,
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
    pub running: Arc<AtomicBool>,
    logpathbase: String,
    log: OpLog,
    failure_prob: f64,
    success_prob_ops: f64,
    success_prob_msg: f64,
    tx: Sender<ProtocolMessage>,
    rx: Receiver<ProtocolMessage>,
    committed: i32,
    aborted: i32,
    unknown: i32,
}

// static timeout for receiving final decision from coordinator
static TIMEOUT: Duration = Duration::from_millis(80);

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
        logpathbase: String,
        failure_prob: f64,
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
            logpathbase: logpathbase.clone(),
            log: OpLog::new(format!("{}/participant_{}.log", logpathbase, id)),
            failure_prob,
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
        trace!("{}::send...", self.id_string);
        let result: bool = true;
        self.tx.send(pm).unwrap();
        trace!("{}::send exit", self.id_string);
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
            trace!("{}::send_unreliable failed", self.id_string);
            result = false;
        }
        result
    }

    ///
    /// recv_request()
    /// receive a message from coordinator
    ///
    pub fn recv_request(&self) -> Option<ProtocolMessage> {
        trace!("{}::recv_request...", self.id_string);
        assert!(self.state != ParticipantState::Voting);

        let mut result = None;
        let received = self.rx.recv();
        if let Ok(pm) = received {
            info!("{}  received {:?}", self.id_string, pm);
            result = Some(pm);
        }
        trace!("{}::recv_request exit", self.id_string);
        result
    }

    ///
    /// recv_decision()
    /// receive a final decision from coordinator
    ///
    pub fn recv_decision(&self, txid: i32) -> ProtocolMessage {
        trace!("{}::recv_decision...", self.id_string);
        assert!(self.state == ParticipantState::Confirming);

        let mut result = None;
        let start_time = Instant::now();
        loop {
            if start_time.elapsed() > TIMEOUT {
                info!("{}  receiving timed out", self.id_string);
                break;
            }
            if let Ok(pm) = self.rx.try_recv() {
                if pm.txid == txid {
                    info!("{}  received {:?}", self.id_string, pm);
                    result = Some(pm);
                    break;
                }
            }
        }
        while let None = result {
            let oplog_global = OpLog::from_file(format!("{}/coordinator.log", self.logpathbase));
            for offset in (1..(oplog_global.seqno + 1)).rev() {
                let pm = oplog_global.read(&offset);
                if pm.txid == txid {
                    result = Some(pm.clone());
                    break;
                }
            }
        }
        trace!("{}::recv_decision exit", self.id_string);
        result.unwrap()
    }

    ///
    /// perform_operation
    /// perform the operation specified in the 2PC proposal,
    /// with some probability of success/failure determined by the
    /// command-line option success_probability.
    ///
    pub fn perform_operation(&self) -> bool {
        trace!("{}::perform_operation", self.id_string);
        assert!(self.state == ParticipantState::Voting);

        let mut result = false;

        let x: f64 = random();
        if x < self.success_prob_ops {
            info!("{}  operation succeeded", self.id_string);
            result = true;
        } else {
            info!("{}  operation failed", self.id_string);
        }

        trace!("{}::perform_operation exit", self.id_string);
        result
    }

    ///
    /// report_status()
    /// report the abort/commit/unknown status (aggregate) of all
    /// transaction requests made by this coordinator before exiting.
    ///
    pub fn report_status(&self) {
        println!(
            "{}:\tC:{}\tA:{}\tU:{}",
            self.id_string, self.committed, self.aborted, self.unknown
        );
    }

    ///
    /// protocol()
    /// Implements the participant side of the 2PC protocol
    /// HINT: if the simulation ends early, don't keep handling requests!
    /// HINT: wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self) {
        trace!("{}::protocol", self.id_string);

        let mut txid: i32;
        let mut senderid: String;
        let mut opid: i32;

        self.log = OpLog::from_file(format!("{}/{}.log", self.logpathbase, self.id_string));
        let seqno = self.log.seqno;
        if seqno > 0 {
            // recovery protocol, build internal state from commitlog
            let pm = self.log.read(&seqno);
            info!("{}  recovered from {:?}", self.id_string, pm);
            txid = pm.txid;
            senderid = pm.senderid.clone();
            opid = pm.opid;
            if let MessageType::ParticipantVoteCommit = pm.mtype {
                self.state = ParticipantState::Recovering;
            } else {
                // flush old messsages
                while let Ok(_) = self.rx.try_recv() {}
                self.state = ParticipantState::Quiescent;
            }
        } else {
            // initialize
            txid = -1;
            senderid = String::new();
            opid = -1;
            self.state = ParticipantState::Quiescent;
        }

        loop {
            info!("{}  ({:?})", self.id_string, self.state);
            match self.state {
                ParticipantState::Quiescent => {
                    // get request
                    let request = self.recv_request();
                    if let None = request {
                        panic!("COORDINATOR DISCONNECTED");
                    }
                    let request = request.unwrap();
                    match request.mtype {
                        MessageType::CoordinatorPropose => (),
                        MessageType::CoordinatorExit => break,
                        _ => continue,
                    }
                    txid = request.txid;
                    senderid = request.senderid;
                    opid = request.opid;
                    self.state = ParticipantState::Voting;
                }
                ParticipantState::Voting => {
                    // perform operation
                    let op_success = self.perform_operation();

                    // vote
                    let vote: ProtocolMessage;
                    let mtype = if op_success {
                        MessageType::ParticipantVoteCommit
                    } else {
                        MessageType::ParticipantVoteAbort
                    };
                    vote = ProtocolMessage::generate(mtype, txid, senderid.clone(), opid);
                    self.send_unreliable(vote.clone());
                    self.log.append_pm(vote);
                    self.state = ParticipantState::Confirming;
                }
                ParticipantState::Confirming => {
                    // get final decision, update locally
                    let decision = self.recv_decision(txid);
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
                ParticipantState::Recovering => {
                    // get final decision from global log
                    let mut decision = None;
                    while let None = decision {
                        let oplog_global =
                            OpLog::from_file(format!("{}/coordinator.log", self.logpathbase));
                        for offset in (1..(oplog_global.seqno + 1)).rev() {
                            let pm = oplog_global.read(&offset);
                            if pm.txid == txid {
                                decision = Some(pm.clone());
                                break;
                            }
                        }
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
                    // flush old messsages
                    while let Ok(_) = self.rx.try_recv() {}
                    self.state = ParticipantState::Quiescent;
                }
            }
            let x: f64 = random();
            if x < self.failure_prob {
                info!("{}  FAILURE", self.id_string);
                return;
            }
        }

        self.report_status();

        trace!("{}  exiting", self.id_string);
    }
}
