use ndarray::{Array1, ArrayView1, ArrayView2, s};
use std::collections::{BinaryHeap, HashMap};
use rand::{distributions::Uniform, Rng, prelude::*};
pub use ordered_float::*;
use crate::algs::*;
use crate::util::{DebugTimer};

#[derive(Clone, PartialEq, Debug)]
pub struct Centroid {
    pub point: Array1::<f64>,
    pub children: Vec::<usize>
}

impl Centroid {
    fn new(point: Array1::<f64>) -> Self {
        Centroid {
            point: point,
            children: Vec::<usize>::new()
        }
    }
}

#[derive(Debug, Clone)]
pub struct KMeans {
    name: String,
    metric: String,
    codebook: HashMap::<usize, Centroid>,
    clusters: usize,
    max_iterations: usize,
    clusters_to_search: usize,
    verbose_print: bool
}

impl KMeans {
    pub fn new(verbose_print: bool, clusters: usize, max_iterations: usize, clusters_to_search: usize) -> Self {
        KMeans {
            name: "FANN_kmeans()".to_string(),
            metric: "angular".to_string(),
            codebook: HashMap::<usize, Centroid>::new(),
            clusters: clusters,
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbose_print: verbose_print
        }
    }

    fn init(&mut self, dataset: &ArrayView2::<f64>) {
        let mut rng = thread_rng();
        let dist_uniform = Uniform::new_inclusive(0, dataset.nrows());
        for i in 0..self.clusters {
            let rand_key = rng.sample(dist_uniform);
            let candidate = dataset.slice(s![rand_key,..]);
            let new_centroid = Centroid::new(candidate.to_owned());
            self.codebook.insert(i, new_centroid);
        }
    }

    fn assign(&mut self, dataset: &ArrayView2::<f64>) {
        // Delete points associated to each centroid
        for (_, centroid) in self.codebook.iter_mut() {
            centroid.children.clear();
        }
        for (idx, candidate) in dataset.outer_iter().enumerate() {
            let mut best_centroid = 0;
            let mut best_distance = f64::NEG_INFINITY;
            for (&key, centroid) in self.codebook.iter_mut() {
                let distance = distance::cosine_similarity(&(centroid.point).view(), &candidate);
                if OrderedFloat(best_distance) < OrderedFloat(distance) {
                    best_centroid = key;
                    best_distance = distance;
                }
            }
            self.codebook.get_mut(&best_centroid).unwrap().children.push(idx);       
        }
    }

    fn update(&mut self, dataset: &ArrayView2::<f64>) {
        for (_, centroid) in self.codebook.iter_mut() {
            for i in 0..centroid.point.len() {
                centroid.point[i] = 0.;
            }
            
            for child_key in centroid.children.iter() {
                let child_point = dataset.slice(s![*child_key,..]);
                for (i, x) in child_point.iter().enumerate() {
                    centroid.point[i] += x;
                }
            }

            for i in 0..centroid.point.len() {
                centroid.point[i] = centroid.point[i]/centroid.children.len() as f64;
            }
        }
    }
    
    fn run_kmeans(&mut self, max_iterations: usize, dataset: &ArrayView2::<f64>) {
        self.init(dataset);
        // Repeat until convergence or some iteration count
        let mut last_codebook: HashMap::<usize, Centroid> = HashMap::new();
        let mut iterations = 1;
        loop {
            if iterations > max_iterations {
                if self.verbose_print {
                    println!("Max iterations reached, iterations: {}", iterations-1);
                }
                break;
            } else if self.codebook == last_codebook {
                if self.verbose_print {
                    println!("Computation has converged, iterations: {}", iterations-1);
                }
                break;
            }
    
            last_codebook = self.codebook.clone();

            self.assign(dataset);
            self.update(dataset);
            iterations += 1;
        }
    }
}

impl AlgorithmImpl for KMeans {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        self.run_kmeans(self.max_iterations, &dataset);
    }

    fn query(&self, dataset: &ArrayView2::<f64>, p: &ArrayView1::<f64>, result_count: usize) -> Vec<usize> {        
        let mut best_centroids = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();

        for (key, centroid) in self.codebook.iter() {
            let distance = distance::cosine_similarity(&p, &centroid.point.view());
            if best_centroids.len() < self.clusters_to_search as usize {
                best_centroids.push((OrderedFloat(-distance), *key));
            } else {
                if OrderedFloat(distance) > -best_centroids.peek().unwrap().0 {
                    best_centroids.pop();
                    best_centroids.push((OrderedFloat(-distance), *key));
                }
            }
        }
        
        let mut best_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        for _ in 0..self.clusters_to_search {
            let centroid_key = best_centroids.pop().unwrap().1;
            for candidate_key in self.codebook.get(&(centroid_key)).unwrap().children.iter() {
                let candidate = dataset.slice(s![*candidate_key,..]);
                let distance = distance::cosine_similarity(&p, &candidate);
                if best_candidates.len() < result_count as usize {
                    best_candidates.push((OrderedFloat(-distance), *candidate_key));
                } else {
                    if OrderedFloat(distance) > -best_candidates.peek().unwrap().0 {
                        best_candidates.pop();
                        best_candidates.push((OrderedFloat(-distance), *candidate_key));
                    }
                }
            }
        }
    
        let mut best_n_candidates: Vec<usize> = Vec::new();
        for _ in 0..best_candidates.len() {
            best_n_candidates.push(best_candidates.pop().unwrap().1);
        }
        best_n_candidates.reverse();
        best_n_candidates
    }
}
