extern crate ndarray;
extern crate hdf5;
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

fn main() {
    let dataset_name = "glove-100-angular";
    let run_count = 1;
    let result_count: u32 = 10;
    let distance_type = "cosine";
    let build_time = 0.; // Not used
    let index_size = 0.; // Not used

    // let algo_definition = "bruteforce";
    // let alg_name = "bruteforce_basic";

    let algo_definition = "kmeans";
    let alg_name = "kmeans_basic";

    let best_search_time = f64::INFINITY;

    let filename = format!("datasets/{}.hdf5",dataset_name);
    let ds = Dataset::new(&filename);
    let ds_train_norm = ds.train_normalize();
    let ds_test_norm = ds.test_normalize();
    let ds_distances_norm = ds.distances_normalize();
    let ds_neighbors = ds.neighbors();

    ds.print_true_neighbors(0, 5, 10);

    let dataset = &ds_test_norm;
    let mut results = Vec::<(f64, Vec<(usize, f64)>)>::new();

    let mut algo = &algs::get_algorithm(&ds_train_norm.view());


    for (i, p) in dataset.outer_iter().enumerate() {
    // let p = &ds_test_norm.slice(s![0,..]);
        let result = algs::run_individual_query(algo, &p, &ds_train_norm.view(), result_count);
        println!("{:?}", result);
        // break;
        results.push(result);
        // if i > 5 {break}
    }
    return;
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
        index_size: index_size,
        algo: algo_definition.to_string(),
        dataset: dataset_name.to_string(),

        batch_mode: false,
        best_search_time: best_search_time,
        candidates: avg_candidates,
        count: result_count,
        distance: distance_type.to_string(),
        expect_extra: false,
        name: alg_name.to_string(),
        run_count: run_count
    };

    store_results_and_fix_attributes(results, attrs);

    println!("Hello there");
   
}

