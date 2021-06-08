use ndarray::{ArrayView1, ArrayView2, s};
use std::collections::{BinaryHeap};
use rand::{prelude::*};
use ordered_float::*;
use crate::algs::*;
use crate::algs::{kmeans::{kmeans}, common::{Centroid}};
use crate::util::{DebugTimer};
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;
use bincode::serialize_into;

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
                        name: "fa_kmeans_REF_0608_0950".to_string(),
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
        let fit_from_file = true;
        let file_fa_kmeans_codebook = "saved_objects/fa_kmeans_codebook.bin";

        // Load existing pre-computede data if exists
        if fit_from_file && Path::new(file_fa_kmeans_codebook).exists() 
                                && Path::new(file_fa_kmeans_codebook).exists() {
            let mut t = DebugTimer::start("fit fa_kmeans_codebook from file");
            let mut read_file = File::open(file_fa_kmeans_codebook).unwrap();
            self.codebook = bincode::deserialize_from(&mut read_file).unwrap();
            t.stop();
            t.print_as_secs();
        } else {
            // Write compute_coarse_quantizers to bin
            let mut t = DebugTimer::start("Fit write fa_kmeans_codebook to file");
            let mut new_file = File::create(file_fa_kmeans_codebook).unwrap();
            serialize_into(&mut new_file, &self.codebook).unwrap();
            t.stop();
            t.print_as_secs();
        }
    }

    fn query(&self, dataset: &ArrayView2::<f64>, query: &ArrayView1::<f64>, results_per_query: usize, arguments: &Vec::<usize>) -> Vec<usize> { 
        // Query Arguments
        let clusters_to_search = arguments[0];      
        // let best_centroids = &mut BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();

        // for centroid in self.codebook.iter() {
        //     let distance = distance::cosine_similarity(query, &centroid.point.view());
        //     best_centroids.push((OrderedFloat(distance), *&centroid.id));
        // }

        let mut best_centroids: BinaryHeap::<(OrderedFloat::<f64>, usize)> = self.codebook.par_iter().map(|centroid| {
            let distance = distance::cosine_similarity(query, &centroid.point.view());
            (OrderedFloat(distance), *&centroid.id)
        }).collect();

        let clusters_of_interests: Vec<usize> = (0..std::cmp::min(clusters_to_search, best_centroids.len())).map(|_| best_centroids.pop().unwrap().1).collect();

        let best_candidates = &mut BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        for centroid_index in clusters_of_interests.iter() {

            for candidate_key in self.codebook[*centroid_index].indexes.iter() {
                let candidate = dataset.slice(s![*candidate_key,..]);
                let neg_distance = OrderedFloat(-distance::cosine_similarity(query, &candidate.view()));
                // If candidates list is shorter than min results requestes push to heap
                if best_candidates.len() < results_per_query {
                    best_candidates.push((neg_distance, *candidate_key));
                }
                // If distance is better, remove top (worst) and push candidate to heap
                if neg_distance < best_candidates.peek().unwrap().0 {
                    best_candidates.pop();
                    best_candidates.push((neg_distance, *candidate_key));
                }
            }
        }

        // Pop all candidate indexes from heap and reverse list.
        let mut best_n_candidates: Vec<usize> =  (0..best_candidates.len()).map(|_| best_candidates.pop().unwrap().1).collect();
        best_n_candidates.reverse();
        best_n_candidates
    }
}
