pub mod bruteforce;
pub mod kmeans;
pub mod dataset;
pub mod distance;
pub mod pq;
use std::time::{Instant, Duration};
use ndarray::{ArrayBase, ArrayView1, ArrayView2, Array2};
use ndarray::{s};
use bruteforce::Bruteforce;

pub fn single_query(p: &ArrayView1<f64>, dataset: &ArrayView2<f64>, result_count: u32) -> (f64, Vec<(usize, f64)>) {
    let time_start = Instant::now();
    // let alg = Bruteforce::new("bruteforce");
    let alg = Bruteforce::new();
    // let dataset = Array2::from(vec![vec![1;1];1]);
    alg.fit(*dataset);
    let candidates = alg.query(&p, result_count);
    // let candidates = bruteforce::query(&p, &dataset, result_count);
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

trait AlgorithmImpl {

    fn new() -> Self;

    fn done(&self);

    fn get_memory_usage(&self);

    fn fit(&mut self, dataset: ArrayView2::<f64>);

    fn query(&self, p: &ArrayView1::<f64>, result_count: u32) -> Vec<usize>;

    fn batch_query(&self);

    fn get_batch_results(&self);
    
    fn get_additional(&self);

    fn __str__(&self);
}