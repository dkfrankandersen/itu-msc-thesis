use ndarray::{Array1, ArrayView1, ArrayView2, s};
use crate::algs::*;


#[derive(Clone, PartialEq, Debug)]
pub struct Centroid {
    id: i32,
    pub point: Array1::<f64>,
    pub children: Vec::<usize>
}

#[allow(dead_code)]
impl Centroid {
    fn new(id: i32, point: Array1::<f64>) -> Self {
        Centroid {
            id: id,
            point: point,
            children: Vec::<usize>::new()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scann {
    name: String,
    metric: String,
    dataset: Option<Array2::<f64>>,
    clusters: i32,
    max_iterations: i32,
    clusters_to_search: i32,
    verbose_print: bool
}


impl Scann {
    pub fn new(verbose_print: bool, dataset: &ArrayView2::<f64>, clusters: i32, max_iterations: i32, clusters_to_search: i32) -> Self {
        Scann {
            name: "FANN_scann()".to_string(),
            metric: "angular".to_string(),
            dataset: None,
            clusters: clusters,
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbose_print: verbose_print
        }
    }
}

impl AlgorithmImpl for Scann {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn fit(&mut self, dataset: &ArrayView2::<f64>) {
        self.dataset = Some(dataset.to_owned());
    }

    fn query(&self, dataset: &ArrayView2::<f64>, p: &ArrayView1::<f64>, result_count: usize) -> Vec<usize> {
        
        let mut best_n_candidates: Vec<usize> = Vec::new();
        best_n_candidates.reverse();
        if self.verbose_print {
            println!("best_n_candidates \n{:?}", best_n_candidates);
        }
        best_n_candidates
    }

}