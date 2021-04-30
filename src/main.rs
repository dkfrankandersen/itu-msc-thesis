use std::env;
mod algs;
use algs::dataset::Dataset;
mod util;
use util::*;
mod running;


fn main() {
    let verbose_print = true;
    let args: Vec<String> = env::args().collect();
    
    println!("Running algorithm with");
    println!("args: {:?}\n", args);
    let mut para_results: Option::<Vec::<usize>> = None;
    let mut para_arguments: Option::<String> = None;
    let mut para_query: Option::<String> = None;

    if args.len() >= 4 {
        let args_additionals = args[4..].join(" ");
        let parts = unzip_enclosed_text(args_additionals, '[', ']');
        if parts.len() >= 1 { 
            para_results = Some(parts[0].split_whitespace().map(|x| (x.to_string()).parse::<usize>().unwrap()).collect());
        };
        if parts.len() >= 2 { para_arguments = Some(parts[1].to_string())};
        if parts.len() >= 3 { para_query = Some(parts[2].to_string()); }
    } else {
        println!("Arguments missing, should be [metric dataset algorithm results] [algs optionals] [query optionals]");
        return;
    }
    let para = if para_arguments.is_some() {para_arguments.unwrap().split_whitespace().map(|x| x.to_string()).collect()} else {Vec::<String>::new()};
    let parameters = util::RunParameters{ 
                                    metric: args[1].to_string(), 
                                    dataset: args[2].to_string(),
                                    algorithm: args[3].to_string(),
                                    additional: para,
    };

    fn create_run_parameters(metric: String, dataset: String, algorithm: String, results: Vec<usize>, arguments: Vec::<String>, query_options: Vec::<String>) {
        
    } 

    let best_search_time = f64::INFINITY;
    let filename = format!("datasets/{}.hdf5",parameters.dataset);
    let ds = Dataset::new(&filename);
    let ds_train_norm = ds.train_normalize();
    let ds_test_norm = ds.test_normalize();
    // let ds_distances_norm = ds.distances_normalize();
    // let ds_neighbors = ds.neighbors();
    
    // if verbose_print {
    //     ds.print_true_neighbors(0, 5, 10);
    // }


    
    let dataset = &ds_test_norm;
    let (build_time, algo) = algs::get_fitted_algorithm(verbose_print, &parameters.algorithm, parameters.additional.clone(), &ds_train_norm.view());

    let mut clusters_to_search: Vec::<usize> = Vec::<usize>::new();
    if para_query.is_some(){
        clusters_to_search = para_query.unwrap().split_whitespace().map(|x| (x.to_string()).parse::<usize>().unwrap()).collect();
    }
    // println!("Start running individual querys");
    for results_per_query in para_results.unwrap().iter() {
        let mut results = Vec::<(f64, Vec<(usize, f64)>)>::new();
        if clusters_to_search.len() > 0 {
            for cluster_to_search in clusters_to_search.iter() {
                let query_arguments: Vec::<usize> = vec![*cluster_to_search];
                for (_, p) in dataset.outer_iter().enumerate() {
                    let result = algs::run_individual_query(&algo, &p, &ds_train_norm.view(), *results_per_query, &query_arguments);
                    results.push(result);
                }
                running::compute_timing_and_store(best_search_time, build_time, results.clone(), *results_per_query, dataset.nrows(), parameters.clone());
            }
        } else {
            let empty_args = &Vec::<usize>::new();
            for (_, p) in dataset.outer_iter().enumerate() {
                let result = algs::run_individual_query(&algo, &p, &ds_train_norm.view(), *results_per_query, empty_args);
                results.push(result);
            }
            running::compute_timing_and_store(best_search_time, build_time, results.clone(), *results_per_query, dataset.nrows(), parameters.clone());
        }


        
    }
}