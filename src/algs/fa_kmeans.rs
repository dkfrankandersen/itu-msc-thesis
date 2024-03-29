use ndarray::{ArrayView1, ArrayView2, s};
use std::collections::{BinaryHeap};
use rand::{prelude::*};
use ordered_float::*;
use crate::algs::{AlgorithmImpl, AlgoParameters};
use crate::algs::distance::{DistanceMetric, min_distance};
use crate::algs::{kmeans::{KMeans}, common::{Centroid}};
use crate::util::{debug_timer::DebugTimer};
use std::fs::File;
use std::path::Path;
use indicatif::ProgressBar;
use bincode::serialize_into;
use std::io::{BufWriter, BufReader};

#[derive(Debug, Clone)]
pub struct FAKMeans {
    name: String,
    metric: String,
    algo_parameters: AlgoParameters,
    codebook: Vec<Centroid>,
    k_clusters: usize,
    max_iterations: usize,
    verbose_print: bool,
    dist_metric: DistanceMetric
}

impl FAKMeans {
    pub fn new(verbose_print: bool, dist_metric: DistanceMetric, algo_parameters: &AlgoParameters, k_clusters: usize, max_iterations: usize) -> Result<Self, String> {
        if k_clusters == 0 {
            return Err("Clusters must be greater than 0".to_string());
        }
        else if max_iterations == 0 {
            return Err("max_iterations must be greater than 0".to_string());
        }

        Ok(
            FAKMeans {
                        name: "fa_kmeans".to_string(),
                        metric: algo_parameters.metric.clone(),
                        algo_parameters: algo_parameters.clone(),
                        codebook: Vec::<Centroid>::new(),
                        k_clusters,
                        max_iterations,
                        verbose_print,
                        dist_metric
                    })
    }
}

impl AlgorithmImpl for FAKMeans {

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {

        let file_fa_kmeans_codebook = &self.algo_parameters.fit_file_output("codebook");

        // Load existing pre-computede data if exists
        if Path::new(file_fa_kmeans_codebook).exists()
                && Path::new(file_fa_kmeans_codebook).exists() {
            let mut t = DebugTimer::start("fit fa_kmeans_codebook from file");
            let mut read_file = BufReader::new(File::open(file_fa_kmeans_codebook).unwrap());
            self.codebook = bincode::deserialize_from(&mut read_file).unwrap();
            t.stop();
            t.print_as_millis();
        } else {
            // Write compute_coarse_quantizers to bin
            let rng = thread_rng();
            let mut t = DebugTimer::start("fit run kmeans");
            let bar_max_iterations = ProgressBar::new((self.max_iterations) as u64);
            let kmeans = KMeans::new(&self.dist_metric);
            self.codebook = kmeans.run(rng, self.k_clusters, self.max_iterations, dataset, false, &bar_max_iterations);
            bar_max_iterations.finish();
            t.stop();
            t.print_as_millis();

            println!("\nFit write centroids to file: {}", file_fa_kmeans_codebook);
            let mut t = DebugTimer::start("Fit write fa_kmeans_codebook to file");
            let mut new_file = BufWriter::new(File::create(file_fa_kmeans_codebook).unwrap());
            serialize_into(&mut new_file, &self.codebook).unwrap();
            t.stop();
            t.print_as_secs();
        }
    }

    fn query(&self, dataset: &ArrayView2::<f64>, query: &ArrayView1::<f64>, results_per_query: usize, arguments: &[usize]) -> Vec<usize> {
        // Query Arguments
        let clusters_to_search = arguments[0];

        // Calculate distance between query and all centroids, collect result into max heap
        let mut query_centroid_distances: BinaryHeap::<(OrderedFloat::<f64>, usize)> = self.codebook.iter().map(|centroid| {
            (OrderedFloat(-min_distance(query, &centroid.point.view(), &self.dist_metric)), centroid.id)}).collect();

        // Collect best centroid indexes, limit by clusters_to_search
        let min_val = std::cmp::min(clusters_to_search, query_centroid_distances.len());
        let best_centroid_indexes: Vec<usize> = (0..min_val).map(|_| query_centroid_distances.pop().unwrap().1).collect();

        // For every best centroid, collect best candidates with negative distance into max heap, and limited heap size by replacing worst with better.
        let best_candidates = &mut BinaryHeap::<(OrderedFloat::<f64>, usize)>::new();
        for centroid_index in best_centroid_indexes.into_iter() {
            for candidate_key in self.codebook[centroid_index].indexes.iter() {
                let candidate_index = *candidate_key;
                let candidate = dataset.slice(s![*candidate_key,..]);

                let distance = OrderedFloat(min_distance(&candidate, query, &self.dist_metric));

                // If candidates list is shorter than min results requestes push to heap
                if best_candidates.len() < results_per_query {
                    best_candidates.push((distance, candidate_index));
                }
                // If distance is better, remove top (worst) and push candidate to heap
                else if distance < best_candidates.peek().unwrap().0 {
                    best_candidates.pop();
                    best_candidates.push((distance, candidate_index));
                }
            }
        }

        // Pop all candidate indexes from heap and reverse list.
        let mut best_n_candidates: Vec<usize> =  (0..best_candidates.len())
                                                    .map(|_| best_candidates
                                                    .pop().unwrap().1).collect();
        best_n_candidates.reverse();
        best_n_candidates
    }
}
