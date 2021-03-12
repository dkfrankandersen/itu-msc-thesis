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

#[derive(Debug, Clone)]
pub enum Algorithm {
    Bruteforce(Bruteforce),
    KMeans(KMeans),
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
            Algorithm::Bruteforce(ref mut bruteforce) => bruteforce.fit(dataset),
            Algorithm::KMeans(ref mut kmeans) => kmeans.fit(dataset),
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
    pub fn get(algorithm: &str, args: Vec<String>) -> Algorithm {
        match algorithm.as_ref() {
            "bruteforce" => Algorithm::Bruteforce(Bruteforce::new()),
            "kmeans" => Algorithm::KMeans(KMeans::new(args[0].parse::<i32>().unwrap(), args[1].parse::<i32>().unwrap(), args[2].parse::<i32>().unwrap())),
            &_ => unimplemented!(),
        }
    }
}


pub fn get_fitted_algorithm(algo: &str, args: Vec<String>, dataset: &ArrayView2<f64>) -> (f64, Algorithm) {
    
    let mut algo = AlgorithmFactory::get(algo, args);

    println!("Starting dataset fitting for algorithm");
    let time_start = Instant::now();
    algo.fit(dataset.view());
    let time_finish = Instant::now();
    let total_time = time_finish.duration_since(time_start);
    println!("Timespend in algorithm fitting: {}s", total_time.as_secs());
    
    (total_time.as_secs_f64(), algo)
}

pub fn run_individual_query(algo: &Algorithm, p: &ArrayView1<f64>, dataset: &ArrayView2<f64>, result_count: u32) -> (f64, Vec<(usize, f64)>) {
    let time_start = Instant::now();
    let candidates = algo.query(&p, result_count);
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