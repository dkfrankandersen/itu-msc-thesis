use ndarray::{Array, Array1, Array2, s, ArrayView1};
use std::collections::{HashMap};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Centroid {
    pub id: usize,
    pub point: Array1<f64>,
    pub indexes: Vec::<usize>
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PQCentroid {
    pub id: usize,
    pub point: Array1<f64>,
    pub children: HashMap::<usize, Vec::<usize>>
}