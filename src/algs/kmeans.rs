use ndarray::{ArrayBase, Array, Array1, Array2, ArrayView1, ArrayView2, s};
use crate::algs::distance;
use crate::algs::pq;
use std::collections::BinaryHeap;
use rand::prelude::*;
use std::collections::HashMap;
use std::cmp::Ordering;
use colored::*;

#[derive(Clone, PartialEq, Debug)]
pub struct Centroid {
    id: i32,
    pub point: Array1::<f64>,
    pub children: Vec::<usize>
}

impl Centroid {
    fn new(id: i32, point: Array1::<f64>) -> Centroid {
        Centroid {
            id: id,
            point: point,
            children: Vec::<usize>::new()
        }
    }
}

pub fn kmeans(k: i32, max_iterations: i32, max_samples: i32, dataset: &ArrayView2::<f64>) -> HashMap::<i32, Centroid> {
    /*
        Repeat X times, select best based on cluster density {
            Repeat until convergence or some iteration count {
                Init
                Assign
                Update
            }
            Save result
        }

        Pick best result
    */

    let n = &dataset.shape()[0]; // shape of rows, cols (vector dimension)
    let mut rng = thread_rng();
    let dist_uniform = rand::distributions::Uniform::new_inclusive(0, n);
    let mut init_k_sampled: Vec<usize> = vec![];

    // 1. Init
    let mut codebook = HashMap::<i32, Centroid>::new();
    for i in 0..k {
        let rand_key = rng.sample(dist_uniform);
        init_k_sampled.push(rand_key);
        let candidate = dataset.slice(s![rand_key,..]);
        let new_centroid = Centroid::new(i, candidate.to_owned());
        codebook.insert(i, new_centroid);
    }

    println!("Dataset lenght: {}", n);
    println!("Init k-means with centroids: {:?}\n", init_k_sampled);
    print_codebook("Codebook after init", &codebook);

    // Repeat until convergence or some iteration count
    let mut iterations = 1;
    let mut last_codebook = HashMap::new();
    loop {
        if iterations == 1 || iterations % 10 == 0 {
            println!("Iteration {}", iterations);
        }
        

        if iterations > max_iterations {
            println!("Max iterations reached, iterations: {}", iterations-1);
            break;
        } else if codebook == last_codebook {
            println!("Computation has converged, iterations: {}", iterations-1);
            break;
        }

        last_codebook = codebook.clone();

        // Delete points associated to each centroid
        for (_, centroid) in codebook.iter_mut() {
            centroid.children.clear();
        }

        // 2. Assign
        // println!("Let's look for my favorit centroid!");
        for (idx, candidate) in dataset.outer_iter().enumerate() {
            let mut best_centroid = -1;
            let mut best_distance = f64::NEG_INFINITY;
            for (&key, centroid) in codebook.iter_mut() {
                let distance = distance::cosine_similarity(&(centroid.point).view(), &candidate);
                if best_distance < distance {
                    best_centroid = key;
                    best_distance = distance;
                }
            }
            if best_centroid >= 0 {
                codebook.get_mut(&best_centroid).unwrap().children.push(idx);
            }            
        }

        // print_codebook("Codebook after assign", &codebook);

        // 3. Update
        for (_, centroid) in codebook.iter_mut() {
                for i in 0..centroid.point.len() {
                    centroid.point[i] = 0.;
                }
                
                for child_key in centroid.children.iter() {
                    let child_point = dataset.slice(s![*child_key,..]);
                    for (i, x) in child_point.iter().enumerate() {
                        centroid.point[i] += x;
                    }
                }

                let dimensions = centroid.point.len() as f64;
                for i in 0..centroid.point.len() {
                    centroid.point[i] = centroid.point[i]/dimensions;
                }
        }
        // print_codebook("Codebook after update", &codebook);
        iterations += 1;
    }
    print_sum_codebook_children("Does codebook contain all points?", &codebook, *n);
    codebook
}

pub fn query(p: &ArrayView1::<f64>, dataset: &ArrayView2::<f64>, result_count: u32) -> Vec<usize> {
    
    let codebook = kmeans(10, 200, 10, dataset);
    let centroids_to_search = 1;
    let mut best_centroids = BinaryHeap::new();
    let mut best_candidates = BinaryHeap::new();
    for (key, centroid) in codebook.iter() {
        best_centroids.push(pq::DataEntry {
            index: *key as usize,  
            distance: distance::cosine_similarity(&p, &centroid.point.view())
        });
    }


    for i in 0..centroids_to_search {
        let centroid_key = best_centroids.pop().unwrap().index;
        for candidate_key in codebook.get(&(centroid_key as i32)).unwrap().children.iter() {
            let candidate = dataset.slice(s![centroid_key as i32,..]);
            best_candidates.push(pq::DataEntry {
                                                    index: *candidate_key,  
                                                    distance: distance::cosine_similarity(&p, &candidate)
                                                });
        }
    }

    

    let mut best_n_candidates: Vec<usize> = Vec::new();
    for _ in 0..result_count {
        let idx = (Some(best_candidates.pop()).unwrap()).unwrap();
        best_n_candidates.push(idx.index);
    }
    println!("best_n_candidates \n{:?}", best_n_candidates);
    best_n_candidates
}

fn print_sum_codebook_children(info: &str, codebook: &HashMap<i32, Centroid>, dataset_len: usize) {
    println!("{}", info.to_string().on_white().black());
    let mut sum = 0;
    for (_, centroid) in codebook.iter() {
        sum += centroid.children.len();
    }
    println!("children: {} == {} dataset points, equal {}", sum.to_string().blue(), dataset_len.to_string().blue(), (sum == dataset_len).to_string().blue());
}

fn codebook_simple_hash(info: &str, codebook: &HashMap<i32, Centroid>) {
    for (key, centroid) in codebook.iter() {
        println!("-> centroid C{:?} |  children: {:?} | point sum: {:?}", key, centroid.children.len(), centroid.point.sum());
    }
}

fn print_codebook(info: &str, codebook: &HashMap<i32, Centroid>) {
    println!("{}", info.to_string().on_white().black());
    for (key, centroid) in codebook.iter() {
        println!("-> centroid C{:?} |  children: {:?} | point sum: {:?}", key, centroid.children.len(), centroid.point.sum());
    }
}