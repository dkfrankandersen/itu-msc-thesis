pub mod bruteforce;
pub mod dataset;
pub mod distance;
pub mod pq;
use std::time::{Instant, Duration};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::{s};

pub fn single_query(p: &ArrayView1<f64>, dataset: &ArrayView2<f64>) -> (f64, Vec<(usize, f64)>) {
    let time_start = Instant::now();
    let candidates = bruteforce::query(&p, &dataset, 10);
    let time_finish = Instant::now();
    let total_time = time_finish.duration_since(time_start);

    let mut candidates_dist: Vec<(usize, f64)> = Vec::new();
    for i in candidates.iter() {
        let q = &dataset.slice(s![*i as i32,..]);
        let dist = distance::cosine_similarity(p, q);
        candidates_dist.push((*i, dist));
    }

    (total_time.as_secs_f64(), candidates_dist)
}