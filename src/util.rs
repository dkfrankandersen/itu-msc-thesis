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
#[derive(Clone, Debug)]
pub struct DebugTimer {
    msg: String,
    start: Option<Instant>,
    stop: Option<Instant>
}

#[allow(dead_code)]
impl DebugTimer {
    pub fn start(msg: &str) -> Self {
        DebugTimer {
            msg: msg.to_string(),
            start: Some(Instant::now()),
            stop: None
        }
    }

    pub fn stop(&mut self) {
        self.stop = Some(Instant::now());
    }

    pub fn print_as_secs(&self) {
        let duration =  self.stop.unwrap().duration_since(self.start.unwrap()).as_secs();
        println!("TimerDebug:  {} in {} s", self.msg, duration);
    }

    pub fn print_as_millis(&self) {
        let duration =  self.stop.unwrap().duration_since(self.start.unwrap()).as_millis();
        println!("TimerDebug:  {} in {} ms", self.msg, duration);
    }


}

pub fn unzip_enclosed_text(text: String, start: char, end: char) -> Vec::<String> {
    let mut pairs = Vec::<(i32, i32)>::new();
    for (i, c) in text.chars().enumerate() {
        if c == start { pairs.push((i as i32,-1)); }
        if c == end {
            for j in (0..pairs.len()).rev() {
                if pairs[j].1 == -1 {
                    pairs[j] = (pairs[j].0, i as i32);
                    break;
                }
            }
        }
    }
    let mut results = Vec::<String>::new();
    let mut max: i32 = i32::MIN;
    for (a, b) in pairs.iter() {
        if a > &max {
            max = *b;
            let r = &text[(*a as usize)+1..(*b as usize)];
            results.push(r.to_string());
        }
    }
    results
}
