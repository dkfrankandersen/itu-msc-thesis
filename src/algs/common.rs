use ndarray::{Array1, ArrayView1};
use std::collections::{HashMap};
use std::collections::{BinaryHeap};
use crate::algs::distance;
use ordered_float::*;

#[derive(Clone, PartialEq, Debug)]
pub struct Centroid {
    pub id: usize,
    pub point: Array1<f64>,
    pub indexes: Vec::<usize>
}

impl Centroid {
    pub fn new(id: usize, point: Array1::<f64>) -> Self {
        Centroid {
            id: id,
            point: point,
            indexes: Vec::<usize>::new()
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct PQCentroid {
    pub id: usize,
    pub point: Array1<f64>,
    pub children: HashMap::<usize, Vec::<usize>>
}

pub fn push_to_max_cosine_heap(heap: &mut BinaryHeap::<(OrderedFloat::<f64>, usize)>, query: &ArrayView1::<f64>, 
                                centroid_point: &ArrayView1::<f64>, centroid_index: &usize, minimum_clusters: usize) {
    let distance = distance::cosine_similarity(query, centroid_point);
    if heap.len() < minimum_clusters {
        heap.push((OrderedFloat(-distance), *centroid_index));
    } else {
        if OrderedFloat(distance) > -heap.peek().unwrap().0 {
            heap.pop();
            heap.push((OrderedFloat(-distance), *centroid_index));
        }
    }
}
