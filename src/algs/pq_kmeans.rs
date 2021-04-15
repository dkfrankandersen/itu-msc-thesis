
use ndarray::{Array1, ArrayView2, s};
use rand::prelude::*;
use rand::{Rng};
use colored::*;
use crate::algs::{distance};

#[derive(Debug, Clone)]
pub struct PQKMeans {
    k: usize,
    max_iterations: usize,
    codebook: Vec::<(Array1::<f64>, Vec::<usize>)>,
    verbose_print: bool
}

impl PQKMeans {

    pub fn new(k: usize, max_iterations: usize) -> Self {
        PQKMeans{
            k: k,
            max_iterations: max_iterations,
            codebook: Vec::with_capacity(k),
            verbose_print: false
        }
    }

    fn _print_codebook(&self, info: &str, codebook: &Vec::<(Array1::<f64>, Vec::<usize>)>) {
        println!("{}", info.to_string().on_white().black());
        for (k, (centroid, children)) in codebook.iter().enumerate() {
            println!("-> centroid C{:?} |  children: {:?} | point sum: {:?}", k, children.len(), centroid.sum());
        }
    }

    pub fn run(&mut self, dataset: &ArrayView2::<f64> ) -> &Vec::<(Array1::<f64>, Vec::<usize>)> {

        self.codebook = Vec::with_capacity(self.k);
        self.init(dataset);        
        let mut last_codebook = Vec::with_capacity(self.k);
        let mut iterations = 1;
        loop {
            if self.verbose_print && (iterations == 1 || iterations % 10 == 0) {
                println!("Iteration {}", iterations);
            }
            if iterations > self.max_iterations {
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
        if self.verbose_print {
            self._print_codebook("Codebook after run: ", &self.codebook);
        }
        
        return &self.codebook;
    }

    fn init(&mut self, dataset: &ArrayView2::<f64>) {
        let mut rng = thread_rng();
        let dist_uniform = rand::distributions::Uniform::new_inclusive(0, dataset.nrows()-1);
        for _ in 0..self.k {
            let rand_key = rng.sample(dist_uniform);
            let candidate = dataset.slice(s![rand_key,..]);
            self.codebook.push((candidate.to_owned(), Vec::<usize>::new()));
        }

        if self.verbose_print {
            self._print_codebook("Codebook after init: ", &self.codebook);
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
                // let distance = distance::cosine_similarity(&(centroid.0).view(), &candidate);
                let distance = (centroid.0).view().dot(&candidate);
                if best_distance < distance {
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
    
                for i in 0..centroid.len() {  
                    centroid[i] = centroid[i]/children.len() as f64;
                }
            }
        }
    }


}