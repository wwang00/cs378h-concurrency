#[macro_use]
extern crate log;
extern crate clap;
extern crate ctrlc;
extern crate stderrlog;
use std::thread;
use std::thread::JoinHandle;
pub mod checker;
pub mod client;
pub mod coordinator;
pub mod message;
pub mod oplog;
pub mod participant;
pub mod tpcoptions;
use client::Client;
use coordinator::Coordinator;
use participant::Participant;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::mpsc::channel;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

///
/// register_clients()
///
/// The coordinator needs to know about all clients.
/// This function should create clients and use some communication
/// primitive to ensure the coordinator and clients are aware of
/// each other and able to exchange messages. Starting threads to run the
/// client protocol should be deferred until after all the communication
/// structures are created.
///
/// HINT: you probably want to look at rust's mpsc::channel or crossbeam
///       channels to set up communication. Communication in 2PC
///       is duplex!
///
/// HINT: read the logpathbase documentation carefully.
///
/// <params>
///     coordinator: the coordinator!
///     n_clients: number of clients to create and register
///     logpathbase: each participant, client, and the coordinator
///         needs to maintain its own operation and commit log.
///         The project checker assumes a specific directory structure
///         for files backing these logs. Concretely, participant log files
///         will be expected to be produced in:
///            logpathbase/client_<num>.log
///     running: atomic bool indicating whether the simulation is still running
///
fn register_clients(
    coordinator: &mut Coordinator,
    n_clients: i32,
    running: &Arc<AtomicBool>,
) -> Vec<Client> {
    let mut clients = vec![];
    // register clients with coordinator (set up communication channels and sync objects)
    // add client to the vector and return the vector.
    for c in 0..n_clients {
        let name = format!("client_{}", c);
        let (tx, rx) = coordinator.client_join();
        clients.push(Client::new(c, name, running.clone(), n_clients, tx, rx));
    }
    clients
}

///
/// register_participants()
///
fn register_participants(
    coordinator: &mut Coordinator,
    n_participants: i32,
    running: &Arc<AtomicBool>,
    logpathbase: &String,
    success_prob_ops: f64,
    success_prob_msg: f64,
) -> Vec<Participant> {
    let mut participants = vec![];
    for p in 0..n_participants {
        let name = format!("participant_{}", p);
        let logpath = format!("{}/participant_{}.log", logpathbase, p);
        let (tx, rx) = coordinator.participant_join();
        participants.push(Participant::new(
            p,
            name,
            running.clone(),
            logpath,
            success_prob_ops,
            success_prob_msg,
            tx,
            rx,
        ))
    }
    participants
}

///
/// launch_clients()
///
/// create a thread per client to run the client
///
fn launch_clients(clients: Vec<Client>, n_requests: i32, handles: &mut Vec<JoinHandle<()>>) {
    for mut client in clients {
        let handle = thread::spawn(move || {
            client.protocol(n_requests);
        });
        handles.push(handle);
    }
}

///
/// launch_participants()
///
/// create a thread per participant to run the participant
///
fn launch_participants(participants: Vec<Participant>, handles: &mut Vec<JoinHandle<()>>) {
    for mut participant in participants {
        let handle = thread::spawn(move || {
            participant.protocol();
        });
        handles.push(handle);
    }
}

///
/// run()
/// opts: an options structure describing mode and parameters
///
/// 0. install a signal handler that manages a global atomic boolean flag
/// 1. creates a new coordinator
/// 2. creates new clients and registers them with the coordinator
/// 3. creates new participants and registers them with coordinator
/// 4. launches participants in their own threads
/// 5. launches clients in their own threads
/// 6. creates a thread to run the coordinator protocol
///
fn run(opts: &tpcoptions::TPCOptions) {
    // vector for wait handles, allowing us to
    // wait for client, participant, and coordinator
    // threads to join.
    let mut handles: Vec<JoinHandle<()>> = vec![];

    // create an atomic bool object and a signal handler
    // that sets it. this allows us to inform clients and
    // participants that we are exiting the simulation
    // by pressing "control-C", which will set the running
    // flag to false.
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("CTRL-C!");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting signal handler!");

    // create a coordinator, create and register clients and participants
    // launch threads for all, and wait on handles.
    let mut coordinator: Coordinator;
    let clients: Vec<Client>;
    let participants: Vec<Participant>;

    // init coordinator
    let logpath = format!("{}/coordinator.log", opts.logpath);
    coordinator = Coordinator::new(
        running.clone(),
        logpath,
        opts.success_probability_ops,
        opts.success_probability_msg,
        opts.num_clients,
        opts.num_participants,
    );

    // init clients
    clients = register_clients(&mut coordinator, opts.num_clients, &running);

    // init participants
    participants = register_participants(
        &mut coordinator,
        opts.num_participants,
        &running,
        &opts.logpath,
        opts.success_probability_ops,
        opts.success_probability_msg,
    );

    // launch
    launch_participants(participants, &mut handles);
    launch_clients(clients, opts.num_requests, &mut handles);
    let coordinator_handle = thread::spawn(move || {
        coordinator.protocol();
        trace!("coordinator terminated");
    });
    handles.push(coordinator_handle);

    // wait
    for handle in handles {
        handle.join().unwrap();
    }
}

///
/// main()
///
fn main() {
    let opts = tpcoptions::TPCOptions::new();
    stderrlog::new()
        .module(module_path!())
        .quiet(false)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .verbosity(opts.verbosity)
        .init()
        .unwrap();

    match opts.mode.as_ref() {
        "run" => run(&opts),
        "check" => checker::check_last_run(
            opts.num_clients,
            opts.num_requests,
            opts.num_participants,
            &opts.logpath,
        ),
        _ => panic!("unknown mode"),
    }
}
