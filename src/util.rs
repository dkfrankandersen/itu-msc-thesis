pub mod hdf5_attributes_fix;
pub mod hdf5_store_file;
pub mod testcases;
pub mod dataset;

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

use std::time::{Instant};
#[derive(Clone, Debug)]
pub struct DebugTimer {
    msg: String,
    start: Option<Instant>,
    stop: Option<Instant>
}

#[allow(dead_code)]
impl DebugTimer {
    pub fn start(msg: &str) -> Self {
        DebugTimer {
            msg: msg.to_string(),
            start: Some(Instant::now()),
            stop: None
        }
    }

    pub fn stop(&mut self) {
        self.stop = Some(Instant::now());
    }

    pub fn print_as_secs(&self) {
        let duration =  self.stop.unwrap().duration_since(self.start.unwrap()).as_secs();
        println!("TimerDebug:  {} in {} s", self.msg, duration);
    }

    pub fn print_as_millis(&self) {
        let duration =  self.stop.unwrap().duration_since(self.start.unwrap()).as_millis();
        println!("TimerDebug:  {} in {} ms", self.msg, duration);
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
        let algo_arg = format!("{:?}", self.algo_arguments).to_string().replace(",","").replace('"',"").replace(" ","_");
        let query_arg = format!("{:?}", self.query_arguments).to_string().replace(",","").replace('"',"").replace(" ","_");
        return format!("{}({}_{}_{})", self.algorithm, self.metric, algo_arg, query_arg);
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

pub fn create_run_parameters(args: Vec::<String>) -> AlgoParameters {

    let mut results_per_query = Vec::<usize>::new();
    let mut algo_arguments = Vec::<String>::new();
    let mut query_arguments = Vec::<usize>::new();

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
            query_arguments = parts[2].to_string().split_whitespace().map(|x| (x.to_string()).parse::<usize>().unwrap()).collect()
        }
    } else {
        println!("Arguments missing, should be [metric dataset algorithm results] [algs optionals] [query optionals]");
    }

    let metric = &args[1].to_string();
    let dataset = &args[2].to_string();
    let algorithm = &args[3].to_string();

    let mut run_parameters = Vec::<RunParameters>::new();
    for results_per_query in results_per_query.iter() {
        if query_arguments.len() > 0 {
            for cluster_to_search in query_arguments.iter() {
                let run_parameter = RunParameters{ 
                    metric: metric.to_string(), 
                    dataset: dataset.to_string(),
                    algorithm: algorithm.to_string(),
                    algo_arguments: algo_arguments.clone(),
                    results_per_query: *results_per_query,
                    query_arguments: vec![*cluster_to_search]
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
    return AlgoParameters{metric: metric.to_string(), dataset: dataset.to_string(), algorithm: algorithm.to_string(), 
                                algo_arguments: algo_arguments.clone(), run_parameters: run_parameters};
}