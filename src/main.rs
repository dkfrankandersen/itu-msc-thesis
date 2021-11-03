use std::env;
mod algs;
mod util;
use util::{AlgoParameters, dataset, create_run_parameters};
mod running;
use indicatif::{ProgressBar};
use rayon::iter::{ParallelIterator, IntoParallelRefIterator};
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
    // let ds_distances_norm = ds.distances_normalize();
    // let ds_neighbors = ds.neighbors();
    
    // if verbose_print {
    //     ds.print_true_neighbors(0, 1, 100);
    // }

    let neighbors = vec![
                             // glove-100-angular.hdf5
                            // 97478,  262700,  846101,  671078,  232287,  727732,  544474,
                            // 1133489,  723915,  660281,  566421, 1093917,  908319,  656605,
                            // 93438,  326455,  584887, 1096614,  100206,  547334,  674655,
                            // 834699,  445577,  979282,  776528,   51821,  994865,  186281,
                            // 533888,  331310, 1037752,  193057,  859959,  368655,  690267,
                            // 82685,  484525, 1168162, 1069248, 1126839,  256447,  451625,
                            // 914908,  873104,  956338,  678395,  939324,  748511,  207076,
                            // 751282,  817757,  402216,  932395,  290452,  265744,  696453,
                            // 82910,  436049,  712479,  494528,  989330,  655775,  995275,
                            // 647843,  375237,  242797, 1116578,  793170,  325682,  265226,
                            // 888453,  599119,  631740,  212807, 1142011,  530481,  656064,
                            // 944910,  459704,  490937,  239304,  264602,  495380,  843410,
                            // 724903,  876802,  636623,  172030,  162588,  761652,   74880,
                            // 418892,  687317, 1008844, 1011545,  983601,  340497,  598329,
                            // 944409,  725625

                            // random-xs-20-angular.hdf5
                            3618, 8213, 4462, 6709, 3975, 3129, 5120, 2979, 6319, 3244,  381,
                            5332, 5846,  319, 3325, 1882, 4401, 2044, 7224, 1294, 7539, 5321,
                            3247, 5398,   33, 8582, 7254, 1397, 5700, 4536, 2615, 7802, 3220,
                            4717, 5082, 6604, 2583, 8871, 2275, 4235,  655, 8254, 7007,  511,
                            3502, 4826, 5959,  533, 8705, 8201, 8054, 5335, 7155, 3313, 2820,
                            3974,  185, 5523,  839, 6242, 3192, 2180, 2740, 1477,  992, 3602,
                            3113, 2747, 6137, 5837, 1630,  345, 5159, 8732, 6615, 4195,  325,
                            2969, 8426,  197, 1064, 5957,  647, 1281, 7618, 5121, 6835, 7551,
                            7102, 4981, 6960, 1153, 3357, 1479,  564, 6526, 4545, 6335, 1001,
                            1113
                            ];
    
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
                    // println!("\n\nresult : {:?}\n", &result);
                    // for x in result.1.iter() {
                    //     for (i, y) in neighbors.iter().enumerate() {
                    //         if x.0 == *y {
                    //             println!("Found {} as number {} closest neighbor", x.0, i);
                    //         }
                    //     }
                    // }
                    // panic!("Only query 0 in interessting for now...");
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