
use ndarray::{Array1, ArrayView2, s};
use rand::prelude::*;
extern crate ordered_float;
pub use ordered_float::*;
use crate::util::{sampling::sampling_without_replacement};

#[derive(Debug, Clone)]
pub struct PQResKMeans {
    k: usize,
    max_iterations: usize,
    codebook: Vec::<(Array1::<f64>, Vec::<usize>)>,
    verbose_print: bool
}

impl PQResKMeans {

    pub fn new(k: usize, max_iterations: usize) -> Self {
        PQResKMeans{
            k: k,
            max_iterations: max_iterations,
            codebook: Vec::with_capacity(k),
            verbose_print: false
        }
    }

    pub fn run(&mut self, dataset: &ArrayView2::<f64> ) -> &Vec::<(Array1::<f64>, Vec::<usize>)> {

        self.codebook = Vec::with_capacity(self.k);
        self.init(dataset);        
        let mut last_codebook = Vec::with_capacity(self.k);
        for iterations in 0..self.max_iterations {
            if self.codebook == last_codebook {
                if self.verbose_print {
                    println!("Computation has converged, iterations: {}", iterations);
                }
                break;
            }
    
            last_codebook = self.codebook.clone();

            self.assign(dataset);
            self.update(dataset);
        }
        
        return &self.codebook;
    }

    fn init(&mut self, dataset: &ArrayView2::<f64>) {
        let rng = thread_rng();
        let unique_indexes = sampling_without_replacement(rng, dataset.nrows(), self.k);
        for rand_key in unique_indexes.iter() {
           
            let candidate = dataset.slice(s![*rand_key,..]);
            self.codebook.push((candidate.to_owned(), Vec::<usize>::new()));
        }    
    }

    fn assign(&mut self, dataset: &ArrayView2::<f64>) {
        for (_,children) in self.codebook.iter_mut() {
            children.clear();
        }

        for (idx, candidate) in dataset.outer_iter().enumerate() {
            let mut best_centroid = 0;
            let mut best_distance = f64::NEG_INFINITY;
            for (k, centroid) in self.codebook.iter().enumerate() {
                let distance = (centroid.0).view().dot(&candidate);
                if OrderedFloat(best_distance) < OrderedFloat(distance) {
                    best_centroid = k;
                    best_distance = distance;
                }
            }
            self.codebook[best_centroid].1.push(idx);
        }
    }

    fn update(&mut self, dataset: &ArrayView2::<f64>) {
        for (centroid, children) in self.codebook.iter_mut() {
            if children.len() > 0 {
                for i in 0..centroid.len() {
                    centroid[i]= 0.;
                }
                
                for child_key in children.iter() {
                    let child_point = dataset.slice(s![*child_key,..]);
                    for (i, x) in child_point.iter().enumerate() {
                        centroid[i] += x;
                    }
                }
    
                centroid.mapv_inplace(|a| a/children.len() as f64);
            }
        }
    }


}