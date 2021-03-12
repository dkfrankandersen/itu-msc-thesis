pub mod bruteforce;
pub mod kmeans;
pub mod dataset;
pub mod distance;
pub mod pq;
use std::time::{Instant, Duration};
use ndarray::{ArrayBase, ArrayView1, ArrayView2, Array2};
use ndarray::{s};
use bruteforce::Bruteforce;
use kmeans::KMeans;

#[derive(Debug)]
pub enum Algorithm {
    Bruteforce(Bruteforce),
    KMeans(KMeans),
}

impl AlgorithmImpl for Algorithm {
    
    fn done(&self) {
        match *self {
            Algorithm::Bruteforce(ref bruteforce) => bruteforce.done(),
            Algorithm::KMeans(ref kmeans) => kmeans.done(),
        }
    }

    fn get_memory_usage(&self) {
        match *self {
            Algorithm::Bruteforce(ref bruteforce) => bruteforce.get_memory_usage(),
            Algorithm::KMeans(ref kmeans) => kmeans.get_memory_usage(),
        }
    }

    fn fit(&mut self, dataset: ArrayView2::<f64>) {
        match *self {
            Algorithm::Bruteforce(ref bruteforce) => bruteforce.fit(dataset),
            Algorithm::KMeans(ref kmeans) => kmeans.fit(dataset),
        }
    }

    fn query(&self, p: &ArrayView1::<f64>, result_count: u32) -> Vec<usize> {
        match *self {
            Algorithm::Bruteforce(ref bruteforce) => bruteforce.query(p, result_count),
            Algorithm::KMeans(ref kmeans) => kmeans.query(p, result_count),
        }
    }

    fn batch_query(&self) {
        match *self {
            Algorithm::Bruteforce(ref bruteforce) => bruteforce.__str__(),
            Algorithm::KMeans(ref kmeans) => kmeans.__str__(),
        }
    }

    fn get_batch_results(&self) {
        match *self {
            Algorithm::Bruteforce(ref bruteforce) => bruteforce.__str__(),
            Algorithm::KMeans(ref kmeans) => kmeans.__str__(),
        }
    }
    
    fn get_additional(&self) {
        match *self {
            Algorithm::Bruteforce(ref bruteforce) => bruteforce.get_additional(),
            Algorithm::KMeans(ref kmeans) => kmeans.get_additional(),
        }
    }

    fn __str__(&self) {
        match *self {
            Algorithm::Bruteforce(ref bruteforce) => bruteforce.__str__(),
            Algorithm::KMeans(ref kmeans) => kmeans.__str__(),
        }
    }

    
}

pub struct AlgorithmFactory {}

impl AlgorithmFactory {
    pub fn get(algorithm: &str) -> Algorithm {
        match algorithm.as_ref() {
            "BRUTEFORCE" => Algorithm::Bruteforce(Bruteforce::new()),
            "KMEANS" => Algorithm::KMeans(KMeans::new(10, 200, 1)),
            &_ => unimplemented!(),
        }
    }
}


pub fn get_algorithm(dataset: &ArrayView2<f64>) -> KMeans {
    // let algo = Bruteforce::new("bruteforce");
    println!("Starting dataset fitting for algorithm");
    let mut algo = KMeans::new(10, 200, 1);
    let time_start = Instant::now();
    algo.fit(dataset.view());
    let time_finish = Instant::now();
    let total_time = time_finish.duration_since(time_start);
    println!("Timespend in algorithm fitting: {}s", total_time.as_secs());
    algo
}

pub fn run_individual_query(algo: &KMeans, p: &ArrayView1<f64>, dataset: &ArrayView2<f64>, result_count: u32) -> (f64, Vec<(usize, f64)>) {
    let time_start = Instant::now();
    let candidates = algo.query(&p, result_count);
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

    fn done(&self);

    fn get_memory_usage(&self);

    fn fit(&mut self, dataset: ArrayView2::<f64>);

    fn query(&self, p: &ArrayView1::<f64>, result_count: u32) -> Vec<usize>;

    fn batch_query(&self);

    fn get_batch_results(&self);
    
    fn get_additional(&self);

    fn __str__(&self);
}