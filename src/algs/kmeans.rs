use ndarray::{Array1, ArrayView1, ArrayView2, s};
use crate::algs::distance;
use crate::algs::data_entry::{DataEntry};
use crate::algs::*;
use std::collections::BinaryHeap;
use rand::prelude::*;
use std::collections::HashMap;
use colored::*;

#[derive(Clone, PartialEq, Debug)]
pub struct Centroid {
    id: i32,
    pub point: Array1::<f64>,
    pub children: Vec::<usize>
}

impl Centroid {
    fn new(id: i32, point: Array1::<f64>) -> Self {
        Centroid {
            id: id,
            point: point,
            children: Vec::<usize>::new()
        }
    }
}

#[derive(Debug, Clone)]
pub struct KMeans {
    name: String,
    metric: String,
    dataset: Option<Array2::<f64>>,
    codebook: HashMap::<i32, Centroid>,
    clusters: i32,
    max_iterations: i32,
    clusters_to_search: i32,
    verbode_print: bool
}

impl KMeans {
    pub fn new(verbose_print: bool, clusters: i32, max_iterations: i32, clusters_to_search: i32) -> Self {
        KMeans {
            name: "FANN_bruteforce()".to_string(),
            metric: "cosine".to_string(),
            dataset: None,
            codebook: HashMap::<i32, Centroid>::new(),
            clusters: clusters,
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbode_print: verbose_print
        }
    }

    fn dataset_size(&self) -> usize {
        return self.dataset.as_ref().unwrap().shape()[0]; // shape of rows, cols (vector dimension)
    }

    fn print_sum_codebook_children(&self, info: &str, codebook: &HashMap<i32, Centroid>, dataset_len: usize) {
        println!("{}", info.to_string().on_white().black());
        let mut sum = 0;
        for (_, centroid) in codebook.iter() {
            sum += centroid.children.len();
        }
        println!("children: {} == {} dataset points, equal {}", sum.to_string().blue(), dataset_len.to_string().blue(), (sum == dataset_len).to_string().blue());
    }
    
    fn print_codebook(&self, info: &str, codebook: &HashMap<i32, Centroid>) {
        println!("{}", info.to_string().on_white().black());
        for (key, centroid) in codebook.iter() {
            println!("-> centroid C{:?} |  children: {:?} | point sum: {:?}", key, centroid.children.len(), centroid.point.sum());
        }
    }

    fn init(&mut self, dataset: &ArrayView2::<f64>) {
        let mut rng = thread_rng();
        let dist_uniform = rand::distributions::Uniform::new_inclusive(0, self.dataset_size());
        let mut init_k_sampled: Vec<usize> = vec![];
        for i in 0..self.clusters {
            let rand_key = rng.sample(dist_uniform);
            init_k_sampled.push(rand_key);
            let candidate = dataset.slice(s![rand_key,..]);
            let new_centroid = Centroid::new(i, candidate.to_owned());
            self.codebook.insert(i, new_centroid);
        }

        if self.verbode_print {
            println!("Dataset lenght: {}", self.dataset_size());
            println!("Init k-means with centroids: {:?}\n", init_k_sampled);
            self.print_codebook("Codebook after init", &self.codebook);
        }
    }

    fn assign(&mut self) {
        // Delete points associated to each centroid
        for (_, centroid) in self.codebook.iter_mut() {
            centroid.children.clear();
        }
        for (idx, candidate) in self.dataset.as_ref().unwrap().outer_iter().enumerate() {
            let mut best_centroid = -1;
            let mut best_distance = f64::NEG_INFINITY;
            for (&key, centroid) in self.codebook.iter_mut() {
                let distance = distance::cosine_similarity(&(centroid.point).view(), &candidate);
                if best_distance < distance {
                    best_centroid = key;
                    best_distance = distance;
                }
            }
            if best_centroid >= 0 {
                self.codebook.get_mut(&best_centroid).unwrap().children.push(idx);
            }            
        }
    }

    fn update(&mut self) {
        for (_, centroid) in self.codebook.iter_mut() {
            for i in 0..centroid.point.len() {
                centroid.point[i] = 0.;
            }
            
            for child_key in centroid.children.iter() {
                let child_point = self.dataset.as_ref().unwrap().slice(s![*child_key,..]);
                for (i, x) in child_point.iter().enumerate() {
                    centroid.point[i] += x;
                }
            }

            for i in 0..centroid.point.len() {
                centroid.point[i] = centroid.point[i]/centroid.children.len() as f64;
            }
        }
    }
    
    fn run_kmeans(&mut self, max_iterations: i32, dataset: &ArrayView2::<f64>) {
        self.init(dataset);
        // Repeat until convergence or some iteration count
        let mut iterations = 1;
        let mut last_codebook: HashMap::<i32, Centroid> = HashMap::new();
        loop {
            if self.verbode_print && (iterations == 1 || iterations % 10 == 0) {
                println!("Iteration {}", iterations);
            }
            if iterations > max_iterations {
                if self.verbode_print {
                    println!("Max iterations reached, iterations: {}", iterations-1);
                }
                break;
            } else if self.codebook == last_codebook {
                if self.verbode_print {
                    println!("Computation has converged, iterations: {}", iterations-1);
                }
                break;
            }
    
            last_codebook = self.codebook.clone();

            self.assign();
            self.update();
            iterations += 1;
        }
        if self.verbode_print {
            self.print_sum_codebook_children("Does codebook contain all points?", &self.codebook, dataset.shape()[0]);
            self.print_codebook("Codebook status", &self.codebook);
        }
    }
}

impl AlgorithmImpl for KMeans {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn done(&self) {}

    fn get_memory_usage(&self) {}

    fn fit(&mut self, dataset: ArrayView2::<f64>) {
        self.dataset = Some(dataset.to_owned());
        self.run_kmeans(self.max_iterations, &dataset);
        
    }

    fn batch_query(&self) {}

    fn get_batch_results(&self) {}
    
    fn get_additional(&self) {
        
    }

    fn query(&self, p: &ArrayView1::<f64>, result_count: u32) -> Vec<usize> {
    
        if self.dataset.is_none() {
            println!("Dataset missing");
            return Vec::new();
        }
        
        let mut best_centroids = BinaryHeap::new();
        
        for (key, centroid) in self.codebook.iter() {
            best_centroids.push(DataEntry {
                index: *key as usize,  
                distance: distance::cosine_similarity(&p, &centroid.point.view())
            });
        }
        
        if self.verbode_print {
            println!("best_centroids: {:?}", best_centroids);
        }
        
        let ds = &self.dataset.as_ref().unwrap().view();
        let mut best_candidates = BinaryHeap::new();
        for _ in 0..self.clusters_to_search {
            let centroid_key = best_centroids.pop().unwrap().index;
            for candidate_key in self.codebook.get(&(centroid_key as i32)).unwrap().children.iter() {
                // let neighbors = vec![97478, 262700, 846101, 671078, 232287, 727732, 544474, 1133489, 723915, 660281];
                // if neighbors.contains(candidate_key) {
                //     println!("Best neighbor {:?} is in centroid: {:?}", candidate_key, centroid_key);
                // } 
                let candidate = ds.slice(s![*candidate_key as i32,..]);
                let dist = distance::cosine_similarity(&p, &candidate);
                if best_candidates.len() < result_count as usize {
                    best_candidates.push(DataEntry {
                        index: *candidate_key,  
                        distance: -dist
                    });
                } else {
                    let min_val: DataEntry = *best_candidates.peek().unwrap();
                    if dist > -min_val.distance {
                        best_candidates.pop();
                        best_candidates.push(DataEntry {
                            index: *candidate_key,  
                            distance: -dist
                        });
                    }
                }
            }
        }
    
        let mut best_n_candidates: Vec<usize> = Vec::new();
        for _ in 0..best_candidates.len() {
            let idx = (Some(best_candidates.pop()).unwrap()).unwrap();
            best_n_candidates.push(idx.index);
        }
        best_n_candidates.reverse();
        if self.verbode_print {
            println!("best_n_candidates \n{:?}", best_n_candidates);
        }
        best_n_candidates
    }
}
