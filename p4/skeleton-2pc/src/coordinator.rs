//!
//! coordinator.rs
//! Implementation of 2PC coordinator
//!
extern crate log;
extern crate rand;
extern crate stderrlog;
use coordinator::rand::prelude::*;
use message::MessageType;
use message::ProtocolMessage;
use oplog;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

/// CoordinatorState
/// States for 2PC state machine
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinatorState {
    Quiescent,
    Armed,
    Wait,
    Abort,
    Commit,
}

/// Coordinator
/// struct maintaining state for coordinator
pub struct Coordinator {
    state: CoordinatorState,
    running: Arc<AtomicBool>,
    log: oplog::OpLog,
    success_prob_msg: f64,
    n_clients: i32,
    n_participants: i32,
    tx_clients: Vec<Sender<ProtocolMessage>>,
    rx_clients: Vec<Receiver<ProtocolMessage>>,
    tx_participants: Vec<Sender<ProtocolMessage>>,
    rx_participants: Vec<Receiver<ProtocolMessage>>,
    committed: i32,
    aborted: i32,
    unknown: i32,
}

// static timeout for receiving result from participant
static TIMEOUT: Duration = Duration::from_millis(50);

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
        success_prob_msg: f64,
        n_clients: i32,
        n_participants: i32,
    ) -> Coordinator {
        Coordinator {
            state: CoordinatorState::Quiescent,
            running,
            log: oplog::OpLog::new(logpath),
            success_prob_msg,
            n_clients,
            n_participants,
            tx_clients: Vec::new(),
            rx_clients: Vec::new(),
            tx_participants: Vec::new(),
            rx_participants: Vec::new(),
            committed: 0,
            aborted: 0,
            unknown: 0,
        }
    }

    ///
    /// participant_join()
    /// handle the addition of a new participant
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
    ///
    pub fn client_join(&mut self) -> (Sender<ProtocolMessage>, Receiver<ProtocolMessage>) {
        assert!(self.state == CoordinatorState::Quiescent);
        let (tx_coordinator, rx_client) = channel();
        let (tx_client, rx_coordinator) = channel();
        self.tx_clients.push(tx_coordinator);
        self.rx_clients.push(rx_coordinator);

        (tx_client, rx_client)
    }

    ///
    /// send()
    /// Send a protocol message to the participant.
    /// This variant can be assumed to always succeed.
    ///
    pub fn send(&self, sender: &Sender<ProtocolMessage>, pm: ProtocolMessage) -> bool {
        trace!("coordinator::send...");
        let result: bool = true;
        sender.send(pm).unwrap();
        trace!("coordinator::send exit");
        result
    }

    ///
    /// send_unreliable()
    /// Send a protocol message to the participant,
    /// with some probability of success thresholded by the
    /// command line option success_probability [0.0..1.0].
    ///
    pub fn send_unreliable(&self, sender: &Sender<ProtocolMessage>, pm: ProtocolMessage) -> bool {
        let x: f64 = random();
        let result: bool;
        if x < self.success_prob_msg {
            result = self.send(sender, pm);
        } else {
            result = false;
        }
        result
    }

    ///
    /// recv_request()
    /// receive a message from a client
    ///
    pub fn recv_request(&mut self) -> Option<ProtocolMessage> {
        let mut result = Option::None;
        // trace!("coordinator::recv_request...");
        assert!(self.state == CoordinatorState::Quiescent);

        for c in 0..self.n_clients as usize {
            let rx = &self.rx_clients[c];
            let received = rx.try_recv();
            if let Ok(pm) = received {
                info!("coordinator  received request {:?}", pm);
                result = Some(pm);
                break;
            }
        }
        // trace!("coordinator::recv_request exit");
        result
    }

    ///
    /// collect_votes()
    /// get result of all votes
    ///
    pub fn collect_votes(&mut self, txid: i32) -> bool {
        let mut result = true;
        trace!("coordinator::collect_votes...");
        assert!(self.state == CoordinatorState::Wait);

        for p in 0..self.n_participants as usize {
            info!("coordinator  checking participant_{}", p);
            let rx = &self.rx_participants[p];
            let mut good = false;
            while let Ok(pm) = rx.recv_timeout(TIMEOUT) {
                if pm.txid == txid {
                    good = pm.mtype == MessageType::ParticipantVoteCommit;
                    break;
                }
            }
            if !good {
                result = false;
                break;
            }
        }
        trace!("coordinator::collect_votes exit");
        result
    }

    ///
    /// signal_stop()
    /// tell clients, participants to stop
    ///
    pub fn signal_stop(&self) {
        trace!("coordinator::signal_stop");
        let exit_msg = ProtocolMessage::generate(
            MessageType::CoordinatorExit,
            -1,
            String::from("coordinator"),
            -1,
        );
        for c in 0..self.n_clients as usize {
            let tx = &self.tx_clients[c];
            self.send(tx, exit_msg.clone());
        }
        for p in 0..self.n_participants as usize {
            let tx = &self.tx_participants[p];
            self.send(tx, exit_msg.clone());
        }
    }

    ///
    /// report_status()
    /// report the abort/commit/unknown status (aggregate) of all
    /// transaction requests made by this coordinator before exiting.
    ///
    pub fn report_status(&self) {
        println!(
            "coordinator:\tC:{}\tA:{}\tU:{}",
            self.committed, self.aborted, self.unknown
        );
    }

    ///
    /// protocol()
    /// Implements the coordinator side of the 2PC protocol
    ///
    pub fn protocol(&mut self) {
        trace!("coordinator::protocol");

        while self.running.load(Ordering::SeqCst) {
            // get request
            let request = self.recv_request();
            if let None = request {
                continue;
            }
            let request = request.unwrap();
            let txid = request.txid;
            let senderid = request.senderid;
            let opid = request.opid;
            self.state = CoordinatorState::Armed;

            // send request to participants
            let request_msg = ProtocolMessage::generate(
                MessageType::CoordinatorPropose,
                txid,
                senderid.clone(),
                opid,
            );
            for p in 0..self.n_participants as usize {
                let tx = &self.tx_participants[p];
                self.send_unreliable(tx, request_msg.clone());
            }
            self.state = CoordinatorState::Wait;

            // get participant votes
            let commit = self.collect_votes(txid);
            let mtype: MessageType;
            let state: CoordinatorState;
            if commit {
                self.committed += 1;
                mtype = MessageType::CoordinatorCommit;
                state = CoordinatorState::Commit;
            } else {
                self.aborted += 1;
                mtype = MessageType::CoordinatorAbort;
                state = CoordinatorState::Abort;
            }
            self.log.append(mtype, txid, senderid.clone(), opid);
            self.state = state;

            // send final decision to participants and result to client
            let decision_msg = ProtocolMessage::generate(mtype, txid, senderid.clone(), opid);
            for p in 0..self.n_participants as usize {
                let tx = &self.tx_participants[p];
                self.send_unreliable(tx, decision_msg.clone());
            }
            let client_id = decision_msg.opid % self.n_clients;
            let tx = &self.tx_clients[client_id as usize];
            self.send(tx, decision_msg.clone());
            self.state = CoordinatorState::Quiescent;
        }

        self.signal_stop();
        self.report_status();

        trace!("coordinator  exiting");
    }
}
