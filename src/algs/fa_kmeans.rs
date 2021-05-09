use ndarray::{ArrayView1, ArrayView2, s};
use std::collections::{BinaryHeap};
use rand::{prelude::*};
pub use ordered_float::*;
use crate::algs::*;
use crate::algs::{kmeans::{kmeans}, common::{Centroid}};
//use crate::util::{DebugTimer};

#[derive(Debug, Clone)]
pub struct FAKMeans {
    name: String,
    metric: String,
    codebook: Vec<Centroid>,
    k_clusters: usize,
    max_iterations: usize,
    verbose_print: bool
}

impl FAKMeans {
    pub fn new(verbose_print: bool, k_clusters: usize, max_iterations: usize) -> Result<Self, String> {
        if k_clusters <= 0 {
            return Err("Clusters must be greater than 0".to_string());
        }
        else if max_iterations <= 0 {
            return Err("max_iterations must be greater than 0".to_string());
        }
        
        return Ok(
            FAKMeans {
                        name: "fa_kmeans_C".to_string(),
                        metric: "angular".to_string(),
                        codebook: Vec::<Centroid>::new(),
                        k_clusters: k_clusters,
                        max_iterations: max_iterations,
                        verbose_print: verbose_print
                    });
    }
}

impl AlgorithmImpl for FAKMeans {

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        let rng = thread_rng();
        self.codebook = kmeans(rng, self.k_clusters, self.max_iterations, dataset, false);
    }

    fn query(&self, dataset: &ArrayView2::<f64>, p: &ArrayView1::<f64>, results_per_query: usize, arguments: &Vec::<usize>) -> Vec<usize> { 

        // Query Arguments
        let clusters_to_search = arguments[0];      
        let mut best_centroids = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        for centroid in self.codebook.iter() {
            let distance = distance::cosine_similarity(&p, &centroid.point.view());
            if best_centroids.len() < clusters_to_search {
                best_centroids.push((OrderedFloat(-distance), centroid.id));
            } else {
                if OrderedFloat(distance) > -best_centroids.peek().unwrap().0 {
                    best_centroids.pop();
                    best_centroids.push((OrderedFloat(-distance), centroid.id));
                }
            }
        }

        let mut best_candidates = BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        for _ in 0..clusters_to_search {
            let centroid = best_centroids.pop();
            if centroid.is_some() {
                for candidate_key in self.codebook[centroid.unwrap().1].indexes.iter() {
                    let candidate = dataset.slice(s![*candidate_key,..]);
                    let distance = distance::cosine_similarity(&p, &candidate);
                    if best_candidates.len() < results_per_query {
                        best_candidates.push((OrderedFloat(-distance), *candidate_key));
                    } else {
                        if OrderedFloat(distance) > -best_candidates.peek().unwrap().0 {
                            best_candidates.pop();
                            best_candidates.push((OrderedFloat(-distance), *candidate_key));
                        }
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
