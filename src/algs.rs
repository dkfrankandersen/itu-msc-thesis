pub mod fa_bruteforce;
pub mod fa_kmeans;
pub mod fa_product_quantization;
pub mod kmeans;
pub mod common;
pub mod distance;
pub mod fa_scann;
pub mod scann_kmeans;
pub mod scann_common;

use std::time::{Instant};
use ndarray::{ArrayView1, ArrayView2, s};
use fa_bruteforce::{FABruteforce};
use fa_kmeans::{FAKMeans};
use fa_product_quantization::{FAProductQuantization};
use fa_scann::{FAScann};
use crate::util::{AlgoParameters};
use distance::{DistanceMetric};

#[derive(Debug, Clone)]
pub enum Algorithm {
    FABruteforce(FABruteforce),
    FAKMeans(FAKMeans),
    FAProductQuantization(FAProductQuantization),
    FAScann(FAScann),
}

trait AlgorithmImpl {
    fn fit(&mut self, dataset: &ArrayView2::<f64>);
    fn query(&self, dataset: &ArrayView2::<f64>, p: &ArrayView1::<f64>, results_per_query: usize, arguments: &Vec::<usize>) -> Vec<usize>;
    fn name(&self) -> String;
}

impl AlgorithmImpl for Algorithm {

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        match *self {
            Algorithm::FABruteforce(ref mut x) => x.fit(dataset),
            Algorithm::FAKMeans(ref mut x) => x.fit(dataset),
            Algorithm::FAProductQuantization(ref mut x) => x.fit(dataset),
            Algorithm::FAScann(ref mut x) => x.fit(dataset),
        }
    }

    fn query(&self, dataset: &ArrayView2::<f64>, p: &ArrayView1::<f64>, results_per_query: usize, arguments: &Vec::<usize>) -> Vec<usize> {
        match *self {
            Algorithm::FABruteforce(ref x) => x.query(dataset, p, results_per_query, arguments),
            Algorithm::FAKMeans(ref x) => x.query(dataset, p, results_per_query, arguments),
            Algorithm::FAProductQuantization(ref x) => x.query(dataset, p, results_per_query, arguments),
            Algorithm::FAScann(ref x) => x.query(dataset, p, results_per_query, arguments),
        
        }
    }

    fn name(&self) -> String {
        match *self {
            Algorithm::FABruteforce(ref x) => x.name(),
            Algorithm::FAKMeans(ref x) => x.name(),
            Algorithm::FAProductQuantization(ref x) => x.name(),
            Algorithm::FAScann(ref x) => x.name(),
        }
    }
}
pub struct AlgorithmFactory {}

impl AlgorithmFactory {
    pub fn get(verbose_print: bool, dataset: &ArrayView2::<f64>, algo_parameters: &AlgoParameters) -> Result<Algorithm, String> {
        let dist = DistanceMetric::CosineSimilarity;
        match algo_parameters.algorithm.as_ref() {
            "bruteforce" => {   
                            let alg = FABruteforce::new(verbose_print, dist);
                            match alg {
                                Ok(a) => Ok(Algorithm::FABruteforce(a)),
                                Err(e) => Err(e)
                            }
                            },
            "kmeans" => {
                            let alg = FAKMeans::new(verbose_print, dist, algo_parameters, algo_parameters.algo_arguments[0].parse::<usize>().unwrap(), algo_parameters.algo_arguments[1].parse::<usize>().unwrap());
                            match alg {
                                Ok(a) => Ok(Algorithm::FAKMeans(a)),
                                Err(e) => Err(e)
                            }
                        },
            "pq" => {
                            let alg = FAProductQuantization::new(verbose_print, dist, algo_parameters, dataset, algo_parameters.algo_arguments[0].parse::<usize>().unwrap(), 
                            algo_parameters.algo_arguments[1].parse::<usize>().unwrap(), algo_parameters.algo_arguments[2].parse::<usize>().unwrap(), 
                            algo_parameters.algo_arguments[3].parse::<usize>().unwrap(), algo_parameters.algo_arguments[4].parse::<usize>().unwrap());
                            match alg {
                                Ok(a) => Ok(Algorithm::FAProductQuantization(a)),
                                Err(e) => Err(e)
                            }
                        },
            "scann" => {
                        let alg = FAScann::new(verbose_print, dist, algo_parameters, dataset, algo_parameters.algo_arguments[0].parse::<usize>().unwrap(), 
                        algo_parameters.algo_arguments[1].parse::<usize>().unwrap(), algo_parameters.algo_arguments[2].parse::<usize>().unwrap(), 
                        algo_parameters.algo_arguments[3].parse::<usize>().unwrap(), algo_parameters.algo_arguments[4].parse::<usize>().unwrap(), 
                        algo_parameters.algo_arguments[5].parse::<f64>().unwrap());
                        match alg {
                            Ok(a) => Ok(Algorithm::FAScann(a)),
                            Err(e) => Err(e)
                        }
                    },
            &_ => unimplemented!(),
        }
    }
}

pub fn get_fitted_algorithm(verbose_print: bool, mut algo_parameters: AlgoParameters, dataset: &ArrayView2<f64>) -> Result<(f64, Algorithm, AlgoParameters), String> {
    
    let algo = AlgorithmFactory::get(verbose_print, dataset, &algo_parameters);
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

pub fn run_individual_query(algo: &Algorithm, query: &ArrayView1<f64>, dataset: &ArrayView2<f64>, results_per_query: usize, arguments: &Vec<usize>) -> (f64, Vec<(usize, f64)>) {
    let time_start = Instant::now();
    let candidates = algo.query(dataset, &query, results_per_query, arguments);
    let time_finish = Instant::now();
    let total_time = time_finish.duration_since(time_start);
    // let dist = distance::Distance::;
    let mut candidates_dist: Vec<(usize, f64)> = Vec::new();
    for i in candidates.into_iter() {
        let datapoint = &dataset.slice(s![i,..]);
        let dist = distance::cosine_similarity(query, datapoint);
        candidates_dist.push((i, 1.-dist));
    }

    (total_time.as_secs_f64(), candidates_dist)
}