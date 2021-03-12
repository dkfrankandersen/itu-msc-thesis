extern crate ndarray;
extern crate hdf5;
use std::env;
use std::time::{Instant, Duration};
use ndarray::{s};
mod algs;
use algs::dataset::Dataset;
mod util;
use util::{store_results_and_fix_attributes, hdf5_store_file};

// struct AlgoRun {
//     dataset: String,    // "glove-100-angular"
//     run_count: u32,
//     result_count: u32,
//     distance_type: String,  // "cosine"
//     algo_definition: String, // Bruteforce, k
//     alg_name: String
// }

struct RunParameters {
    metric: String,
    dataset: String,
    algorithm: String,
    results: u32,
    additional: Vec<String>
}

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("args: {:?}", args);

    let parameters = RunParameters{ 
                                    metric: args[1].to_string(), 
                                    dataset: args[2].to_string(),
                                    algorithm: args[3].to_string(),
                                    results: args[3].parse::<u32>().unwrap(),
                                    additional: args[4..].to_vec()
    };

    let run_count = 1;
    let alg_name = "kmeans_basic";

    let best_search_time = f64::INFINITY;

    let filename = format!("datasets/{}.hdf5",parameters.dataset);
    let ds = Dataset::new(&filename);
    let ds_train_norm = ds.train_normalize();
    let ds_test_norm = ds.test_normalize();
    // let ds_distances_norm = ds.distances_normalize();
    // let ds_neighbors = ds.neighbors();

    ds.print_true_neighbors(0, 5, 10);

    let dataset = &ds_test_norm;
    let (build_time, algo) = algs::get_fitted_algorithm("KMEANS", parameters.additional, &ds_train_norm.view());
    let mut results = Vec::<(f64, Vec<(usize, f64)>)>::new();

    for (_, p) in dataset.outer_iter().enumerate() {
        let result = algs::run_individual_query(&algo, &p, &ds_train_norm.view(), parameters.results);
        println!("{:?}", result);
        // break;
        results.push(result);
    }

    let mut total_time: f64 = 0.;
    let mut total_candidates: usize = 0;
    for (time, candidates) in results.iter() {
        total_time += time;
        total_candidates += candidates.len();
    }

    let search_time = total_time / dataset.len() as f64;
    let avg_candidates = total_candidates as f64 / dataset.len() as f64;
    let best_search_time = { if best_search_time < search_time { best_search_time } else { search_time }} ;

    let attrs = hdf5_store_file::Attributes {
        build_time: build_time,
        index_size: 0.,
        algo: parameters.algorithm,
        dataset: parameters.dataset,

        batch_mode: false,
        best_search_time: best_search_time,
        candidates: avg_candidates,
        count: parameters.results,
        distance: parameters.metric,
        expect_extra: false,
        name: alg_name.to_string(),
        run_count: run_count
    };

    store_results_and_fix_attributes(results, attrs);

    println!("Hello there");
   
}

