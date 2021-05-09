use ndarray::{Array1};
use std::collections::{HashMap};

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