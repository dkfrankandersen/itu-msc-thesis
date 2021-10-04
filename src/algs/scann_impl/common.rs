use serde::{Serialize, Deserialize};
use ndarray::prelude::*;
// use ndarray_linalg::*;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Centroid {
    pub id: usize,
    pub point: Array1<f64>,
    pub indexes: Vec::<usize>
}