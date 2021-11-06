use crate::util::{sampling::sampling_without_replacement};
use crate::util::debug_timer::DebugTimer;
use crate::algs::{common::Centroid};
use crate::algs::distance::{DistanceMetric, min_distance};
extern crate sys_info;
use ndarray::{Array, ArrayView2, s};
use rand::{prelude::*};
use ordered_float::*;
use indicatif::ProgressBar;
use std::collections::HashMap;
use std::thread;
use std::sync::Arc;
use rayon::prelude::*;
use crate::algs::scann_common::debug_track_query_top_results;


pub struct KMeans {
    dist_metric: DistanceMetric
}

impl KMeans {
    pub fn new(dist_metric: &DistanceMetric) -> Self {
        KMeans {
            dist_metric: dist_metric.clone()
        }
    }

    pub fn run<T: RngCore>(&self, rng: T, k_centroids: usize, max_iterations: usize,
                    dataset: &ArrayView2::<f64>, verbose_print: bool, bar_max_iterations: &ProgressBar) -> Vec::<Centroid> {
        let datapoint_dimension = dataset.ncols();

        // Init
        let unique_indexes = sampling_without_replacement(rng, dataset.nrows(), k_centroids);
        let mut centroids: Vec::<Centroid> = unique_indexes.into_par_iter()
                    .enumerate()
                    .map(|(k, index)| {
                        let datapoint = dataset.slice(s![index,..]);
                        Centroid{id: k, point: datapoint.to_owned(), indexes: Vec::<usize>::new()}
                    }).collect();

        // Repeat
        let mut last_centroids = Vec::<Centroid>::with_capacity(k_centroids);
        let dataset_arc = Arc::new(dataset.to_owned());
                
        let no_of_threads: usize = sys_info::cpu_num().unwrap_or(1) as usize;
        let max_val = dataset.nrows();
        let chunk = max_val/no_of_threads;

        let mut chunks = Vec::<(usize,usize)>::new();
        for i in 0..no_of_threads {
            let from = chunk*i;
            let mut to = from+chunk;
            if max_val-to < chunk {
                to = max_val;
            }
            chunks.push((from, to));
        }
        
        for iterations in 0..max_iterations  {
            if centroids == last_centroids {
                if verbose_print { println!("Computation has converged, iterations: {}", iterations); }
                break;
            }
            
            last_centroids = centroids.clone();
            
            // Remove centroid children
            centroids.par_iter_mut().for_each(|c| c.indexes.clear());
            
            // Assign     
            let centroids_arc = Arc::new(centroids.clone());
            let dist_metric_arc = Arc::new(self.dist_metric.clone());

            let mut handles = Vec::new();
            for (f, t) in chunks.clone().into_iter() {
                let centroids_arc = Arc::clone(&centroids_arc);
                let dataset_arc = Arc::clone(&dataset_arc);
                let dist_metric_arc = Arc::clone(&dist_metric_arc);
                
                handles.push(thread::spawn(move || {
                    let mut hmap = HashMap::<usize, Vec::<usize>>::new();
                    for index in f..t {
                        let mut best_distance: OrderedFloat::<f64> = OrderedFloat(f64::INFINITY);
                        let mut best_index: usize = 0;
                        let datapoint = &dataset_arc.slice(s![index,..]);
                        for (centroid_index, centroid) in centroids_arc.iter().enumerate() {
                            let distance = OrderedFloat(min_distance(&centroid.point.view(), datapoint, &dist_metric_arc));
                            if distance < best_distance { 
                                best_distance = distance;
                                best_index = centroid_index; 
                            }
                        }
                        if hmap.get(&best_index).is_some() {
                            hmap.get_mut(&best_index).unwrap().push(index);
                        } else {
                            hmap.insert(best_index, vec!(index));
                        }
                    }
                    hmap
                }));
            }

            for handle in handles {
                // Wait for the thread to finish. Returns a result.
                let hmap = handle.join();
                match hmap {
                    Ok(mut hm) => {
                        for (key, val) in hm.iter_mut() {
                            centroids[*key].indexes.append(val);
                        }
                    }
                    Err(e) => {panic!("kmeans threading failed {:?}", e)}
                }
            }
            
            // Update
            for centroid in centroids.iter_mut() {
                if centroid.indexes.len() > 0 {

                    // Clear centroid point
                    centroid.point = Array::from_elem(datapoint_dimension, 0.);
                    
                    // Add dimension value of each
                    for index in centroid.indexes.iter() {
                        let point = dataset.slice(s![*index,..]);
                        for (i, x) in point.iter().enumerate() {
                            centroid.point[i] += x;
                        }
                    }

                    // Divide by indexes to get mean
                    let centroid_indexes_count = centroid.indexes.len() as f64;
                    for i in 0..datapoint_dimension {  
                        centroid.point[i] /= centroid_indexes_count;
                    }
                }
            }
            bar_max_iterations.inc(1);
        }
        centroids
    }
}