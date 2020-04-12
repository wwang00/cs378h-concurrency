//!
//! YOU SHOULD NOT NEED TO CHANGE CODE IN THIS FILE.
//!
extern crate bincode;
extern crate commitlog;
extern crate serde;
extern crate serde_json;
use self::commitlog::message::MessageSet;
use self::commitlog::CommitLog;
use self::commitlog::LogOptions;
use self::commitlog::ReadLimit;
use message;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

pub struct OpLog {
    seqno: i32,
    log_arc: Arc<Mutex<HashMap<i32, message::ProtocolMessage>>>,
    lf: CommitLog,
}

impl OpLog {
    pub fn new(fpath: String) -> OpLog {
        let l = HashMap::new();
        let lck = Mutex::new(l);
        let arc = Arc::new(lck);
        OpLog {
            seqno: 0,
            log_arc: arc,
            lf: CommitLog::new(LogOptions::new(fpath)).unwrap(),
        }
    }
    pub fn from_file(fpath: String) -> OpLog {
        trace!("OpLog::from_file({})", fpath);
        let seqno = 0;
        let mut l = HashMap::new();
        let tlf = CommitLog::new(LogOptions::new(fpath)).unwrap();
        let messages = tlf.read(0, ReadLimit::max_bytes(1 << 20)).unwrap();
        for msg in messages.iter() {
            let line = String::from_utf8(msg.payload().to_vec()).unwrap();
            let pm = message::ProtocolMessage::from_string(&line);
            // info!("{:?}", pm);
            l.insert(pm.uid, pm);
        }
        let lck = Mutex::new(l);
        let arc = Arc::new(lck);
        OpLog {
            seqno: seqno,
            log_arc: arc,
            lf: tlf,
        }
    }
    pub fn append(&mut self, t: message::MessageType, tid: i32, sender: String, op: i32) {
        let lck = Arc::clone(&self.log_arc);
        let mut log = lck.lock().unwrap();
        self.seqno += 1;
        let id = self.seqno;
        let pm = message::ProtocolMessage::generate(t, tid, sender, op);
        let json = serde_json::to_string(&pm).unwrap();
        self.lf.append_msg(json.as_bytes()).unwrap();
        log.insert(id, pm);
    }
    pub fn read(&self, offset: &i32) -> message::ProtocolMessage {
        let lck = Arc::clone(&self.log_arc);
        let log = lck.lock().unwrap();
        let pm = log[&offset].clone();
        pm
    }
    pub fn arc(&self) -> Arc<Mutex<HashMap<i32, message::ProtocolMessage>>> {
        Arc::clone(&self.log_arc)
    }
}
