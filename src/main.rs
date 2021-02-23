extern crate ndarray;
extern crate hdf5;
use std::time::{Instant, Duration};
use ndarray::{s};
mod algs;
use algs::dataset::Dataset;
mod util;

fn main() {
    let dataset_name = "glove-100-angular";
    let filename = format!("datasets/{}.hdf5",dataset_name);
    let ds = Dataset::new(&filename);
    let ds_train_norm = ds.train_normalize();
    let ds_test_norm = ds.test_normalize();
    let ds_distances_norm = ds.distances_normalize();
    let ds_neighbors = ds.neighbors();

    ds.print_true_neighbors(0, 5, 10);


    let dataset = ds_test_norm;
    let count: u32 = 10;

    let mut results = Vec::<(f64, std::vec::Vec<(usize, f64)>)>::new();
    for p in dataset.outer_iter() {
    // let v = &ds_test_norm.slice(s![0,..]);
        let result = algs::single_query(&p, &ds_train_norm.view(), count);
        println!("{:?}", result);
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
    let best_search_time = search_time;
    

    let attrs = util::Attributes {
        batch_mode: false,
        best_search_time: best_search_time,
        build_time: 0.02,
        candidates: avg_candidates,
        count: count,
        dataset: dataset_name.to_string(),
        distance: "cosine".to_string(),
        expect_extra: false,
        index_size: 0.03,
        name: "bruteforce()".to_string(),
        run_count: 3
    };

    let saved = util::store_results(results, attrs);
    println!("{:?}", saved);
}

