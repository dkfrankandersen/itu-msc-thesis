pub mod bruteforce;
pub mod kmeans;
pub mod dataset;
pub mod distance;
pub mod pq;
use std::time::{Instant, Duration};
use ndarray::{ArrayBase, ArrayView1, ArrayView2};
use ndarray::{s};

pub fn single_query(p: &ArrayView1<f64>, dataset: &ArrayView2<f64>, result_count: u32) -> (f64, Vec<(usize, f64)>) {
    let time_start = Instant::now();
    let candidates = bruteforce::query(&p, &dataset, result_count);
    // let candidates = kmeans::query(&p, &dataset, result_count);
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


trait Algorithm {
    fn new(name: &'static str) -> Self;
    fn name(&self) -> &'static str;
    fn definition(&self) -> &'static str;

    fn query(&self, p: &ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 1]>>, 
                    dataset: ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 2]>>, result_count: u32) -> Vec<usize>;
}