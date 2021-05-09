pub mod bruteforce;
pub mod kmeans;
pub mod pq_residuals_kmeans;
pub mod pq_kmeans;
pub mod pq_common;
pub mod product_quantization;
pub mod scann;
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
            Algorithm::Bruteforce(ref x) => x.name(),
            Algorithm::KMeans(ref x) => x.name(),
            Algorithm::ProductQuantization(ref x) => x.name(),
            Algorithm::Scann(ref x) => x.name(),
        }
    }
}

pub struct AlgorithmFactory {}

impl AlgorithmFactory {
    pub fn get(verbose_print: bool, dataset: &ArrayView2::<f64>, algorithm: &str, args: &Vec<String>) -> Result<Algorithm, String> {
        println!("args {:?}", args);
        match algorithm.as_ref() {
            "bruteforce" => {   let alg = Bruteforce::new(verbose_print);
                                match alg {
                                    Ok(a) => Ok(Algorithm::Bruteforce(a)),
                                    Err(e) => Err(e)
                                }
                            },
            "kmeans" => {
                                let alg = KMeans::new(verbose_print, args[0].parse::<usize>().unwrap(), args[1].parse::<usize>().unwrap());
                                match alg {
                                    Ok(a) => Ok(Algorithm::KMeans(a)),
                                    Err(e) => Err(e)
                                }
                        },
            "pq" => {
                        let alg = ProductQuantization::new(verbose_print, dataset, args[0].parse::<usize>().unwrap(), 
                                                                            args[1].parse::<usize>().unwrap(), args[2].parse::<usize>().unwrap(), 
                                                                            args[3].parse::<usize>().unwrap(), args[4].parse::<usize>().unwrap());
                        match alg {
                            Ok(a) => Ok(Algorithm::ProductQuantization(a)),
                            Err(e) => Err(e)
                        }
                        },
            "scann" => {
                        let alg = Scann::new(verbose_print, dataset, args[0].parse::<i32>().unwrap(), 
                                                    args[1].parse::<i32>().unwrap(), args[2].parse::<i32>().unwrap());
                        match alg {
                            Ok(a) => Ok(Algorithm::Scann(a)),
                            Err(e) => Err(e)
                        }
            },
            &_ => unimplemented!(),
        }
    }
}

pub fn get_fitted_algorithm(verbose_print: bool, mut algo_parameters: AlgoParameters, dataset: &ArrayView2<f64>) -> Result<(f64, Algorithm, AlgoParameters), String> {
    
    let algo = AlgorithmFactory::get(verbose_print, dataset, &algo_parameters.algorithm, &algo_parameters.algo_arguments);
    match algo {
        Ok(mut a) => {
                    for elem in algo_parameters.run_parameters.iter_mut() {
                        elem.algorithm = a.name();
                    }
                    
                    println!("Starting dataset fitting for algorithm");
                    let time_start = Instant::now();
                    a.fit(&dataset);
                    let time_finish = Instant::now();
                    let total_time = time_finish.duration_since(time_start);
                    println!("Timespend in algorithm fitting: {}s", total_time.as_secs());
                    return Ok((total_time.as_secs_f64(), a, algo_parameters))
                }
        Err(e) => Err(e)
    }
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