use ndarray::{Array,ArrayView2, s};
use rand::{prelude::*};
pub use ordered_float::*;
use crate::util::{sampling::sampling_without_replacement};
use crate::algs::{distance::euclidian, common::Centroid};
use indicatif::ProgressBar;
use crate::util::debug_timer::DebugTimer;
use std::collections::HashMap;
use std::thread;
use std::sync::Arc;
use rayon::prelude::*;

pub fn scann_kmeans<T: RngCore>(rng: T, k_centroids: usize, max_iterations: usize, dataset: &ArrayView2::<f64>, verbose_print: bool) -> Vec::<Centroid> {
        
    let datapoint_dimension = dataset.ncols();

    // Init
    // let mut centroids = Vec::<Centroid>::with_capacity(k_centroids);
    let unique_indexes = sampling_without_replacement(rng, dataset.nrows(), k_centroids);

    println!("Started scann_kmeans Init");
    let mut t = DebugTimer::start("scann_kmeans init");
    let bar_unique_indexes = ProgressBar::new(unique_indexes.len() as u64);
    let mut centroids: Vec::<Centroid> = unique_indexes.into_par_iter().enumerate().map(|(k, index)| {
        let datapoint = dataset.slice(s![index,..]);
        Centroid{id: k, point: datapoint.to_owned(), indexes: Vec::<usize>::new()}
    }).collect();
    bar_unique_indexes.finish();
    t.stop();
    t.print_as_millis();

    // Repeat
    println!("Started scann_kmeans Repeat");
    let mut t = DebugTimer::start("scann_kmeans Repeat");
    let bar_max_iterations = ProgressBar::new(max_iterations as u64);
    let mut last_centroids = Vec::<Centroid>::with_capacity(k_centroids);
    let dataset_arc = Arc::new(dataset.to_owned());

    const NTHREADS: usize = 8;
    let max_val = dataset.nrows();
    let chunk = max_val/NTHREADS;

    let mut chunks = Vec::<(usize,usize)>::new();
    for i in 0..NTHREADS {
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
        let mut t = DebugTimer::start("Assign"); 
        let centroids_arc = Arc::new(centroids.clone());
        let mut handles = Vec::new();
        for (f, t) in chunks.clone().into_iter() {
            let centroids_arc = Arc::clone(&centroids_arc);
            let dataset_arc = Arc::clone(&dataset_arc);
            handles.push(thread::spawn(move || {
                let mut hmap = HashMap::<usize, Vec::<usize>>::new();
                for index in f..t {
                    let mut best_distance: OrderedFloat::<f64> = OrderedFloat(f64::NEG_INFINITY);
                    let mut best_index: usize = 0;
                    for (centroid_index, centroid) in centroids_arc.iter().enumerate() {
                        let distance = OrderedFloat(euclidian(&centroid.point.view(), &dataset_arc.slice(s![index,..])));
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
        t.stop();
        t.print_as_millis();

        let mut t = DebugTimer::start("handles"); 
        for handle in handles {
            // Wait for the thread to finish. Returns a result.
            let hmap = handle.join();
            match hmap {
                Ok(mut hm) => {
                    for (key, val) in hm.iter_mut() {
                        centroids[*key].indexes.append(val);
                    }
                }
                Err(e) => {panic!("scann_kmeans threading failed {:?}", e)}
            }
        }
        t.stop();
        t.print_as_millis();
        
        // Update
        let mut t = DebugTimer::start("Update"); 
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
                    centroid.point[i] = centroid.point[i]/centroid_indexes_count;
                }
            }
        }
        t.stop();
        t.print_as_millis();
        println!("");
        bar_max_iterations.inc(1);
    }
    bar_max_iterations.finish();
    t.stop();
    t.print_as_secs();
    centroids
}