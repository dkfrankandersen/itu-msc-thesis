pub mod bruteforce;
pub mod kmeans;
pub mod pq_kmeans;
pub mod product_quantization;
pub mod scann;
pub mod dataset;
pub mod distance;
use std::time::{Instant};
use ndarray::{ArrayView1, ArrayView2, Array2, s};
use bruteforce::{Bruteforce};
use kmeans::{KMeans};
use product_quantization::{ProductQuantization};
use scann::{Scann};
use crate::util::*;


#[derive(Debug, Clone)]
pub enum Algorithm {
    Bruteforce(Bruteforce),
    KMeans(KMeans),
    ProductQuantization(ProductQuantization),
    Scann(Scann),
}

trait AlgorithmImpl {
    fn fit(&mut self, dataset: &ArrayView2::<f64>);
    fn query(&self, dataset: &ArrayView2::<f64>, p: &ArrayView1::<f64>, results_per_query: usize, arguments: &Vec::<usize>) -> Vec<usize>;
    fn name(&self) -> String;
}

impl AlgorithmImpl for Algorithm {

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        match *self {
            Algorithm::Bruteforce(ref mut x) => x.fit(dataset),
            Algorithm::KMeans(ref mut x) => x.fit(dataset),
            Algorithm::ProductQuantization(ref mut x) => x.fit(dataset),
            Algorithm::Scann(ref mut x) => x.fit(dataset),
        }
    }

    fn query(&self, dataset: &ArrayView2::<f64>, p: &ArrayView1::<f64>, results_per_query: usize, arguments: &Vec::<usize>) -> Vec<usize> {
        match *self {
            Algorithm::Bruteforce(ref x) => x.query(dataset, p, results_per_query, arguments),
            Algorithm::KMeans(ref x) => x.query(dataset, p, results_per_query, arguments),
            Algorithm::ProductQuantization(ref x) => x.query(dataset, p, results_per_query, arguments),
            Algorithm::Scann(ref x) => x.query(dataset, p, results_per_query, arguments),
        }
    }

    fn name(&self) -> String {
        match *self {
            Algorithm::Bruteforce(ref x) => return x.name(),
            Algorithm::KMeans(ref x) => x.name(),
            Algorithm::ProductQuantization(ref x) => x.name(),
            Algorithm::Scann(ref x) => x.name(),
        }
    }
}

pub struct AlgorithmFactory {}

impl AlgorithmFactory {
    pub fn get(verbose_print: bool, dataset: &ArrayView2::<f64>, algorithm: &str, args: &Vec<String>) -> Algorithm {
        match algorithm.as_ref() {
            "bruteforce" => Algorithm::Bruteforce(Bruteforce::new(verbose_print)),
            "kmeans" => Algorithm::KMeans(KMeans::new(verbose_print, args[0].parse::<usize>().unwrap(), args[1].parse::<usize>().unwrap())),
            "pq" => Algorithm::ProductQuantization(ProductQuantization::new(verbose_print, dataset, args[0].parse::<usize>().unwrap(), 
                                                                            args[1].parse::<usize>().unwrap(), args[2].parse::<usize>().unwrap(), 
                                                                            args[3].parse::<usize>().unwrap(), args[4].parse::<usize>().unwrap())),
            "scann" => Algorithm::Scann(Scann::new(verbose_print, dataset, args[0].parse::<i32>().unwrap(), args[1].parse::<i32>().unwrap(), args[2].parse::<i32>().unwrap())),
            &_ => unimplemented!(),
        }
    }
}

pub fn get_fitted_algorithm(verbose_print: bool, algo_parameters: &AlgoParameters, dataset: &ArrayView2<f64>) -> (f64, Algorithm) {
    
    let mut algo = AlgorithmFactory::get(verbose_print, dataset, &algo_parameters.algorithm, &algo_parameters.algo_arguments);   
    println!("Starting dataset fitting for algorithm");
    let time_start = Instant::now();
    algo.fit(&dataset);
    let time_finish = Instant::now();
    let total_time = time_finish.duration_since(time_start);
    println!("Timespend in algorithm fitting: {}s", total_time.as_secs());
    (total_time.as_secs_f64(), algo)
}

pub fn run_individual_query(algo: &Algorithm, p: &ArrayView1<f64>, dataset: &ArrayView2<f64>, results_per_query: usize, arguments: &Vec::<usize>) -> (f64, Vec<(usize, f64)>) {
    let time_start = Instant::now();
    let candidates = algo.query(dataset, &p, results_per_query, arguments);
    let time_finish = Instant::now();
    let total_time = time_finish.duration_since(time_start);

    let mut candidates_dist: Vec<(usize, f64)> = Vec::new();
    for i in candidates.iter() {
        let q = &dataset.slice(s![*i,..]);
        let dist = distance::cosine_similarity(p, q);
        candidates_dist.push((*i, 1.-dist));
    }

    (total_time.as_secs_f64(), candidates_dist)
}