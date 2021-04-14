extern crate ndarray;
extern crate hdf5;
use std::env;
// use std::time::{Instant, Duration};
// use ndarray::{s};
mod algs;
use algs::dataset::Dataset;
mod util;
use util::{store_results_and_fix_attributes, hdf5_store_file, testcases};

struct RunParameters {
    metric: String,
    dataset: String,
    algorithm: String,
    results: u32,
    additional: Vec<String>,
}

fn algo_definition(rp: &RunParameters) -> String {
    let mut val: String = "".to_string();
    for (i, v) in rp.additional.iter().enumerate() {
        if i>0 { val.push_str("_"); }
        val.push_str(v);
    }
    return format!("{}({}_{})", rp.algorithm, rp.metric, val);
}


fn main() {
    let verbose_print = true;
    let args: Vec<String> = env::args().collect();
    println!("Running algorithm with");
    println!("args: {:?}\n", args);

    let parameters = RunParameters{ 
                                    metric: args[1].to_string(), 
                                    dataset: args[2].to_string(),
                                    algorithm: args[3].to_string(),
                                    results: args[4].parse::<u32>().unwrap(),
                                    additional: args[5..].to_vec(),
    };
    
    let algo_def = algo_definition(&parameters);
    let best_search_time = f64::INFINITY;
    let filename = format!("datasets/{}.hdf5",parameters.dataset);
    let ds = Dataset::new(&filename);
    let ds_train_norm = ds.train_normalize();
    let ds_test_norm = ds.test_normalize();
    // let ds_train_norm = testcases::get_small_1000_6().dataset_norm;
    // let ds_test_norm = testcases::get_small_1000_6().query_norm;
    // let ds_distances_norm = ds.distances_normalize();
    // let ds_neighbors = ds.neighbors();
    
    if verbose_print {
        ds.print_true_neighbors(0, 5, 10);
    }
    
    let dataset = &ds_test_norm;
    let (build_time, algo) = algs::get_fitted_algorithm(verbose_print, &parameters.algorithm, parameters.additional, &ds_train_norm.view());
    let mut results = Vec::<(f64, Vec<(usize, f64)>)>::new();

    for (i, p) in dataset.outer_iter().enumerate() {
        let result = algs::run_individual_query(&algo, &p, &ds_train_norm.view(), parameters.results);
        results.push(result);

        // Debugging on 5 querys
        if i >= 5 {
            break; // debug
        }
    }
    
    // Debug stuff
    let mut debug_best_res = Vec::<Vec::<usize>>::new();
    for (i, (_, res)) in results.iter().enumerate() {
        debug_best_res.push(Vec::<usize>::new());
        for (index, _) in res.iter() {
            debug_best_res[i].push(*index);
        }
    }

    // println!("#### Expected : {:?}", testcases::get_small_1000_6().best_10_results.row(0));
    println!("#### Found    : {:?}", debug_best_res);
    return; // debug

    let mut total_time: f64 = 0.;
    let mut total_candidates: usize = 0;
    for (time, candidates) in results.iter() {
        total_time += time;
        total_candidates += candidates.len();
    }

    let search_time = total_time / dataset.nrows() as f64;
    let avg_candidates = total_candidates as f64 / dataset.nrows() as f64;
    let best_search_time = { if best_search_time < search_time { best_search_time } else { search_time }};

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
        name: algo_def,
        run_count: 1
    };

    store_results_and_fix_attributes(results, attrs);
}

