use ndarray::{ArrayView1, ArrayView2, s};
use std::collections::{BinaryHeap};
use rand::{prelude::*};
use ordered_float::*;
use crate::algs::*;
use crate::algs::{kmeans::{kmeans}, common::{Centroid, push_to_max_cosine_heap}};
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
                        name: "fa_kmeans".to_string(),
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
        let best_centroids = &mut BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();

        for centroid in self.codebook.iter() {
            push_to_max_cosine_heap(best_centroids, p, &centroid.point.view(), &centroid.id, clusters_to_search);
        }

        let best_candidates = &mut BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        let maximum_clusters_to_search = std::cmp::min(clusters_to_search, best_centroids.len());
        for _ in 0..maximum_clusters_to_search {
            let centroid = best_centroids.pop();

            for candidate_key in self.codebook[centroid.unwrap().1].indexes.iter() {
                let candidate = dataset.slice(s![*candidate_key,..]);
                push_to_max_cosine_heap(best_candidates, p, &candidate.view(), candidate_key, results_per_query);
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
