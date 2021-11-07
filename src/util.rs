pub mod hdf5_attributes_fix;
pub mod hdf5_store_file;
pub mod testcases;
pub mod dataset;
pub mod sampling;
pub mod debug_timer;
use std::fs::create_dir_all;

pub fn store_results_and_fix_attributes(results: Vec<(f64, std::vec::Vec<(usize, f64)>)>, attrs: hdf5_store_file::Attributes) {
    let store_result = hdf5_store_file::store_results(results, attrs);
    match store_result {
        Ok(file) => {
            hdf5_attributes_fix::run(file).ok();
        },
        Err(e) => {
                    println!("{}", e);
                }
    }
}

pub fn unzip_enclosed_text(text: String, start: char, end: char) -> Vec::<String> {
    let mut pairs = Vec::<(i32, i32)>::new();
    for (i, c) in text.chars().enumerate() {
        if c == start { pairs.push((i as i32,-1)); }
        if c == end {
            for j in (0..pairs.len()).rev() {
                if pairs[j].1 == -1 {
                    pairs[j] = (pairs[j].0, i as i32);
                    break;
                }
            }
        }
    }
    let mut results = Vec::<String>::new();
    let mut max: i32 = i32::MIN;
    for (a, b) in pairs.iter() {
        if a > &max {
            max = *b;
            let r = &text[(*a as usize)+1..(*b as usize)];
            results.push(r.to_string());
        }
    }
    results
}

#[derive(Clone, Debug)]
pub struct RunParameters {
    pub metric: String,
    pub dataset: String,
    pub algorithm: String,
    pub results_per_query: usize,
    pub algo_arguments: Vec::<String>,
    pub query_arguments: Vec<usize>,
}

impl RunParameters {
    pub fn algo_definition(&self) -> String {
        let algo_arg = format!("{:?}", self.algo_arguments).replace(",","").replace('"',"").replace(" ","_");
        let query_arg = format!("{:?}", self.query_arguments).replace(",","").replace('"',"").replace(" ","_");
        format!("{}[{}_{}_{}]", self.algorithm, self.metric, algo_arg, query_arg)
    }
}

#[derive(Clone, Debug)]
pub struct AlgoParameters {
    pub metric: String,
    pub dataset: String,
    pub algorithm: String,
    pub algo_arguments: Vec::<String>,
    pub run_parameters: Vec<RunParameters>
} 

impl AlgoParameters {
    fn fit_create_path(&self) -> String {
        let s = format!("fit_file_output/{}/{}", self.algorithm, self.dataset);
        let res = create_dir_all(&s);
        match res {
            Ok(_) => println!("Using path for storing output: {}", s),
            Err(e) => panic!("Error: unable to create path/file \n{}", e)
        };
        s
    }   

    pub fn fit_file_output(&self, additional: &str) -> String {
        let path = self.fit_create_path();
        let s = format!("{}/{}_[{}]_{}.bin", path, self.metric, self.algo_arguments.join("_"), additional);
        s
    }   
}

pub fn create_run_parameters(args: Vec::<String>) -> AlgoParameters {

    let mut results_per_query = Vec::<usize>::new();
    let mut algo_arguments = Vec::<String>::new();
    let mut query_arguments = Vec::<String>::new();

    if args.len() >= 4 {
        let args_additionals = args[4..].join(" ");
        let parts = unzip_enclosed_text(args_additionals, '[', ']');
        if parts.len() >= 1 { 
            results_per_query = parts[0].split_whitespace().map(|x| (x.to_string()).parse::<usize>().unwrap()).collect();
        };
        if parts.len() >= 2 { 
            algo_arguments = parts[1].to_string().split_whitespace().map(|x| x.to_string()).collect()
        };
        if parts.len() >= 3 { 
            let leaves_reorder_to_search = unzip_enclosed_text(parts[2].to_string(), '[', ']');
            query_arguments = leaves_reorder_to_search; // .to_string().split_whitespace().map(|x| (x.to_string()).parse::<usize>().unwrap()).collect()
        }
    } else {
        println!("Arguments missing, should be [metric dataset algorithm results] [algs optionals] [query optionals]");
    }

    let metric = &args[1].to_string();
    let dataset = &args[2].to_string();
    let algorithm = &args[3].to_string();

    let mut run_parameters = Vec::<RunParameters>::new();
    for results_per_query in results_per_query.iter() {
        if !query_arguments.is_empty() {
            for cluster_to_search in query_arguments.iter() {
                let run_parameter = RunParameters{ 
                    metric: metric.to_string(), 
                    dataset: dataset.to_string(),
                    algorithm: algorithm.to_string(),
                    algo_arguments: algo_arguments.clone(),
                    results_per_query: *results_per_query,
                    query_arguments: cluster_to_search.to_string().split_whitespace().map(|x| (x.to_string()).parse::<usize>().unwrap()).collect()
                };
                run_parameters.push(run_parameter);
            }
        } else {
            let run_parameter = RunParameters{ 
                metric: metric.to_string(), 
                dataset: dataset.to_string(),
                algorithm: algorithm.to_string(),
                algo_arguments: algo_arguments.clone(),
                results_per_query: *results_per_query,
                query_arguments: Vec::<usize>::new()
            };
            run_parameters.push(run_parameter); 
        }   
        
    }
    AlgoParameters{metric: metric.to_string(), dataset: dataset.to_string(), algorithm: algorithm.to_string(), 
                                algo_arguments, run_parameters}
}