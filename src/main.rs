use std::env;
mod algs;
mod util;
use util::{AlgoParameters, dataset, create_run_parameters};
mod running;
use indicatif::{ProgressBar};
use rayon::iter::{ParallelIterator, IntoParallelRefIterator};
extern crate sys_info;
// use ndarray::{s};

fn print_sys_info() {
    let m = sys_info::mem_info().unwrap();
    println!("+----------------------------------------------+");
    println!("| hostname {:?}", sys_info::hostname().unwrap_or_else(|_| "hostname unknown".to_string()));
    println!("| os_release {:?}", sys_info::os_release().unwrap_or_else(|_| "os_release unknown".to_string()));
    println!("| cpu_speed {:?}", sys_info::cpu_speed().unwrap_or(0));
    println!("| cpu_num {:?}", sys_info::cpu_num().unwrap_or(0));
    println!("| os_type {:?}", sys_info::os_type().unwrap_or_else(|_| "os_type unknown".to_string()));
    println!("| mem_info {:?} GB", m.total/1024/1024);
    println!("+----------------------------------------------+");

}

fn main() {

    print_sys_info();

    let verbose_print = true;
    let args: Vec<String> = env::args().collect();
    
    println!("Running algorithm with");
    println!("args: {:?}\n", args);

    let algo_parameters: AlgoParameters = create_run_parameters(args);

    let best_search_time = f64::INFINITY;
    let filename = format!("datasets/{}.hdf5", algo_parameters.dataset);
    let ds = dataset::Dataset::new(&filename);
    let ds_train_norm = ds.train_normalize();
    let ds_test_norm = ds.test_normalize();
    let ds_neighbors = ds.neighbors();

    // let ds_distances_norm = ds.distances_normalize();
    // let ds_neighbors = dsprint_true_neighbors.neighbors();
    
    // if verbose_print {
    //     ds.(0, 1, 100);
    // }

    // check distance values for test[0] with best
    // let t_idx = 0_usize;
    // let test_point = &ds_test_norm.slice(s![t_idx,..]);
    // println!("{}", test_point);
    // use crate::algs::distance::{cosine_similarity, euclidian};

    // for i in 0..5 {
    //     let n_idx = &ds_neighbors[[t_idx,i]];
    //     let train_point = &ds_train_norm.slice(s![*n_idx,..]).clone();
    //     let dot_dist1 = train_point.dot(test_point);
    //     let euc_dist1 = euclidian(train_point, test_point);
    //     let cos_dist1 = cosine_similarity(train_point, test_point);

    //     println!("{} dot_dist1 {} euc_dist1 {} cos_dist1 {}", n_idx, dot_dist1, euc_dist1, cos_dist1);
    // }
    
    let dataset = &ds_test_norm;
    let algo_fit = algs::get_fitted_algorithm(verbose_print, algo_parameters, &ds_train_norm.view());
    println!();
    match algo_fit {
        Ok(af) => {
            let (build_time, algo, algo_parameters) = af;
            println!("Started running run_parameters {} with {} querys", algo_parameters.run_parameters.len(), dataset.nrows());
            let pb = ProgressBar::new((algo_parameters.run_parameters.len()*dataset.nrows()) as u64);
            algo_parameters.run_parameters.par_iter().for_each(|parameters| {
                let mut results = Vec::<(f64, Vec<(usize, f64)>)>::new();
                for p in dataset.outer_iter() {
                    let result = algs::run_individual_query(&algo, &p, &ds_train_norm.view(), parameters.results_per_query, &parameters.query_arguments);
                    results.push(result);
                    pb.inc(1);
                }
                println!("Store results into HD5F file for {}\n", parameters.algo_definition());
                running::compute_timing_and_store(best_search_time, build_time, results.clone(), parameters.results_per_query, dataset.nrows(), parameters.clone());
            });
            pb.finish();
        },
        Err(e) => eprintln!("{}", e)
    }
}