mod algs;
mod util;
mod running;
use std::env;
use util::{AlgoParameters, dataset, create_run_parameters};
use indicatif::{ProgressBar};
extern crate sys_info;

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
    
    let test_queries = &ds_test_norm;
    let algo_fit = algs::get_fitted_algorithm(verbose_print, algo_parameters, &ds_train_norm.view());
    println!();
    match algo_fit {
        Ok(af) => {
            let (build_time, algo, algo_parameters) = af;
            println!("Started running run_parameters {} with {} querys", algo_parameters.run_parameters.len(), test_queries.nrows());
            let pb = ProgressBar::new((algo_parameters.run_parameters.len()*test_queries.nrows()) as u64);
            algo_parameters.run_parameters.iter().for_each(|parameters| {
                let mut results = Vec::<(f64, Vec<(usize, f64)>)>::new();
                for p in test_queries.outer_iter() {
                    let result = algs::run_individual_query(&algo, &p, &ds_train_norm.view(), parameters.results_per_query, &parameters.query_arguments);
                    results.push(result);
                    pb.inc(1);
                }
                println!("Store results into HD5F file for {}\n", parameters.algo_definition());
                running::compute_timing_and_store(best_search_time, build_time, results.clone(), parameters.results_per_query, test_queries.nrows(), parameters.clone());
            });
            pb.finish();
        },
        Err(e) => eprintln!("{}", e)
    }
}