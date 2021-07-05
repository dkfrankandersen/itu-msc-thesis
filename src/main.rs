use std::env;
mod algs;
mod util;
use util::{AlgoParameters, dataset, create_run_parameters};
mod running;
use indicatif::{ProgressBar};
use rayon::iter::{ParallelIterator, IntoParallelRefIterator};
use algs::fa_scann_util::add_outer_product;
use ndarray::prelude::*;
use ndarray_linalg::Solve;

fn main() {


    // let a: Array2<f64> = random((3, 3));
    // let f = a.factorize_into().unwrap(); // LU factorize A (A is consumed)
    // for _ in 0..10 {
    //     let b: Array1<f64> = random(3);
    //     let x = f.solve_into(b).unwrap(); // Solve A * x = b using factorized L, U
    //     println!("{}", x);
    // }

    let vec: Array2::<f64> = arr2(&[[1., 2., 3.]]);
    // let vec: Array2::<f64> = arr2(&[[1., 2., 3.]]);
    let outer_prodsums: Array2::<f64> = Array2::from_elem((vec.len(), vec.len()), 0.);
    
    // let _assert = add_outer_product(outer_prodsums, vec);
    let val = &vec.t().dot(&vec);
    println!("{:?}", val/(14. as f64));
    panic!("Testing matrix");

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
    // let ds_distances_norm = ds.distances_normalize();
    // let ds_neighbors = ds.neighbors();
    
    // if verbose_print {
    //     ds.print_true_neighbors(0, 1, 100);
    // }
    
    let dataset = &ds_test_norm;
    let algo_fit = algs::get_fitted_algorithm(verbose_print, algo_parameters, &ds_train_norm.view());

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