pub mod hdf5_attributes_fix;
pub mod hdf5_store_file;
pub mod testcases;

pub fn store_results_and_fix_attributes(results: Vec<(f64, std::vec::Vec<(usize, f64)>)>, attrs: hdf5_store_file::Attributes) {
    let store_result = hdf5_store_file::store_results(results, attrs);
    match store_result {
        Ok(file) => {
            hdf5_attributes_fix::run(file).ok();
        },
        Err(e) => {
                    println!("{}", e);
                }
    }
}

use std::time::{Instant};
pub struct TimerDebug {
    msg: String,
    start: Option<Instant>,
    stop: Option<Instant>
}

impl TimerDebug {
    pub fn start(msg: &str) -> Self {
        println!("TimerDebug start: {}", msg);
        TimerDebug {
            msg: msg.to_string(),
            start: Some(Instant::now()),
            stop: None
        }
    }

    pub fn stop_and_print_as_secs(&mut self) {
        self.stop = Some(Instant::now());
        let duration =  self.stop.unwrap().duration_since(self.start.unwrap()).as_secs();
        println!("TimerDebug stop:  {} in {} s", self.msg, duration);
    }

    pub fn stop_and_print_as_millis(&mut self) {
        self.stop = Some(Instant::now());
        let duration =  self.stop.unwrap().duration_since(self.start.unwrap()).as_millis();
        println!("TimerDebug stop:  {} in {} ms", self.msg, duration);
    }
}