pub mod bruteforce;
pub mod dataset;
pub mod distance;
pub mod pq;
use std::time::{Instant, Duration};
use ndarray::{ArrayView1, ArrayView2};

pub fn single_query(p: &ArrayView1<f64>, dataset: &ArrayView2<f64>) -> (Duration, Vec<usize>) {
    let time_start = Instant::now();
    let candidates = bruteforce::query(&p, &dataset, 10);
    let time_finish = Instant::now();
    let total_time = time_finish.duration_since(time_start);
    (total_time, candidates)
}