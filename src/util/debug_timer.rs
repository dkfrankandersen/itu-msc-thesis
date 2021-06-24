use std::time::{Instant, Duration};
#[derive(Clone, Debug)]
pub struct DebugTimer {
    msg: String,
    start: Option<Instant>,
    stop: Option<Instant>,
    stopwatch_durations: Vec<Duration>,
    stopwatch_start: Option<Instant>,
}

#[allow(dead_code)]
impl DebugTimer {
    pub fn start(msg: &str) -> Self {
        DebugTimer {
            msg: msg.to_string(),
            start: Some(Instant::now()),
            stop: None,
            stopwatch_durations: Vec::new(),
            stopwatch_start: None,
        }
    }

    pub fn stop(&mut self) {
        self.stop = Some(Instant::now());
    }

    pub fn total_duration(&self) -> Duration {
        let duration =  self.stop.unwrap().duration_since(self.start.unwrap());
        duration
    }

    pub fn print_as_secs(&self) {
        println!("DebugTimer:  {} in {} s", self.msg, self.total_duration().as_secs());
    }

    pub fn print_as_millis(&self) {
        println!("DebugTimer:  {} in {} ms", self.msg, self.total_duration().as_millis());
    }

    pub fn print_as_nanos(&self) {
        println!("DebugTimer:  {} in {} ns", self.msg, self.total_duration().as_nanos());
    }

    pub fn stopwatch_start(&mut self) {
        self.stopwatch_start = Some(Instant::now());
    }

    pub fn stopwatch_stop(&mut self) {
        self.stopwatch_durations.push(Instant::now().duration_since(self.stopwatch_start.unwrap()));
    }

    pub fn stopwatch_total_duration(&self) -> Duration {
        let mut sum = Duration::new(0, 0);
        for x in self.stopwatch_durations.iter() {
            sum = sum + *x;
        }
        sum
    }

    pub fn print_stopwatch_duration_as_secs(&self) {
        println!("DebugTimer stopwatch:  {} in {} s", self.msg, self.stopwatch_total_duration().as_secs());
    }

    pub fn print_stopwatch_as_millis(&self) {
        println!("DebugTimer stopwatch:  {} in {} ms", self.msg, self.stopwatch_total_duration().as_millis());
    }

    pub fn print_stopwatch_as_nanos(&self) {
        println!("DebugTimer stopwatch:  {} in {} ns", self.msg, self.stopwatch_total_duration().as_nanos());
    }

}