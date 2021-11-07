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
// use crate::algs::scann_common::debug_track_query_top_results;


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
       
        for iterations in 0..max_iterations  {
            if centroids == last_centroids {
                if verbose_print { println!("Computation has converged, iterations: {}", iterations); }
                break;
            }
            
            last_centroids = centroids.clone();
            
            // Remove centroid children
            centroids.par_iter_mut().for_each(|c| c.indexes.clear());

            // Assign
            // let mut point_allocations = vec![0_usize; dataset.len()];
            let mut point_allocations = vec![0_usize; dataset.nrows()];
            point_allocations.par_iter_mut().enumerate().for_each(|(index, assigned_centroid_idx)| {
                let mut best_distance: OrderedFloat::<f64> = OrderedFloat(f64::INFINITY);
                let mut best_centroid_idx: usize = 0;
                let datapoint = &dataset.slice(s![index,..]);
                for (centroid_index, centroid) in centroids.iter().enumerate() {
                    
                    let distance = OrderedFloat(min_distance(&centroid.point.view(), datapoint, &self.dist_metric));
                    if distance < best_distance { 
                        best_distance = distance;
                        best_centroid_idx = centroid_index; 
                    }
                }
                *assigned_centroid_idx = best_centroid_idx;
            });

            for (datapoint_idx, centroid_idx) in point_allocations.iter().enumerate() {
                centroids[*centroid_idx].indexes.push(datapoint_idx);
            }
            
            // Update
            for centroid in centroids.iter_mut() {
                
                // Clear centroid point
                centroid.point = Array::from_elem(datapoint_dimension, 0.);
                
                if centroid.indexes.len() > 0 {
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