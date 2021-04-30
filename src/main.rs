use std::env;
mod algs;
use algs::{dataset::Dataset};
mod util;
use util::*;
mod running;

fn main() {
    let verbose_print = true;
    let args: Vec<String> = env::args().collect();
    
    println!("Running algorithm with");
    println!("args: {:?}\n", args);

    let algo_parameters: AlgoParameters = create_run_parameters(args);

    let best_search_time = f64::INFINITY;
    let filename = format!("datasets/{}.hdf5", algo_parameters.dataset);
    let ds = Dataset::new(&filename);
    let ds_train_norm = ds.train_normalize();
    let ds_test_norm = ds.test_normalize();
    // let ds_distances_norm = ds.distances_normalize();
    // let ds_neighbors = ds.neighbors();
    
    // if verbose_print {
    //     ds.print_true_neighbors(0, 5, 10);
    // }
    
    let dataset = &ds_test_norm;
    let (build_time, algo) = algs::get_fitted_algorithm(verbose_print, &algo_parameters, &ds_train_norm.view());
    // println!("Start running individual querys");
    for parameters in algo_parameters.run_parameters.iter() {
        let mut results = Vec::<(f64, Vec<(usize, f64)>)>::new();
        for (_, p) in dataset.outer_iter().enumerate() {
            let result = algs::run_individual_query(&algo, &p, &ds_train_norm.view(), parameters.results_per_query, &parameters.query_arguments);
            results.push(result);
        }
        running::compute_timing_and_store(best_search_time, build_time, results.clone(), parameters.results_per_query, dataset.nrows(), parameters.clone());
    }
}