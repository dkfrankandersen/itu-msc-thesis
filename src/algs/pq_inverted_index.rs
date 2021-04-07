use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, s};
use std::collections::HashMap;
use crate::algs::*;
use rand::{distributions::Uniform, Rng};
use pq_kmeans::{PQKMeans};
use std::collections::BinaryHeap;
use crate::algs::pq_data_entry::{PQDataEntry};
use crate::algs::data_entry::{DataEntry};


#[derive(Debug, Clone)]
pub struct ProductQuantization {
    name: String,
    metric: String,
    dataset: Option<Array2::<f64>>,
    codebook: Option::<Array2::<Array1::<f64>>>,
    pqcodes: Option::<Array2::<usize>>,
    m: usize,
    training_size: usize,
    k: usize,
    max_iterations: usize,
    clusters_to_search: usize,
    verbose_print: bool,
    dimension: usize,
    sub_dimension: usize
}


impl ProductQuantization {
    pub fn new(verbose_print: bool, m: usize, training_size: usize, k: usize, max_iterations: usize, clusters_to_search: usize) -> Self {
        ProductQuantization {
            name: "FANN_product_quantization()".to_string(),
            metric: "angular".to_string(),
            dataset: None,
            codebook: None,
            pqcodes: None,
            m: m,         // M
            training_size: training_size,
            k: k,         // K
            max_iterations: max_iterations,
            clusters_to_search: clusters_to_search,
            verbose_print: verbose_print,
            dimension: 0,
            sub_dimension: 0
        }
    }

    pub fn random_traindata(&self, dataset: ArrayView2::<f64>, train_dataset_size: usize) -> Array2::<f64> {
        let mut rng = rand::thread_rng();
        let range = Uniform::new(0 as usize, dataset.nrows() as usize);
        let random_datapoints: Vec<usize> = (0..train_dataset_size).map(|_| rng.sample(&range)).collect();
        println!("Random datapoints [{}] for training, between [0..{}]", train_dataset_size, dataset.nrows());
        
        let mut train_data = Array2::zeros((train_dataset_size, dataset.ncols()));
        for (i,v) in random_datapoints.iter().enumerate() {
            let data_row = dataset.slice(s![*v,..]);
            train_data.row_mut(i).assign(&data_row);
        }
        train_data
    }

    fn train_codebook(&mut self, train_data: ArrayView2::<f64>) {
        // Create codebook, [m,k,[sd]] m-th subspace, k-th codewords, sd-th dimension
        let mut codebook = Array::from_elem((self.m, self.k), Array::zeros(self.sub_dimension));
        for m in 0..self.m {
            let begin = self.sub_dimension * m;
            let end = begin + self.sub_dimension - 1;
            let partial_data = train_data.slice(s![.., begin..end]);
            if self.verbose_print {
                // println!("Run k-means for m [{}], sub dim {:?}, first element [{}]", m, partial_data.shape(), partial_data[[0,0]]);
            }
            let mut pq_kmeans = PQKMeans::new(self.k, self.max_iterations);
            let codewords = pq_kmeans.run(partial_data.view());
            for (k, (centroid,_)) in codewords.iter().enumerate() {
                    codebook[[m,k]] = centroid.to_owned();
            }
        }
        self.codebook = Some(codebook);
    }

    struct QuantizerCentroid {
        id: usize,
        childern: Vec::<QuantizedDataPoint>
    }

    struct QuantizedDataPoint {
        index: usize,
        codes: Vec::<usize>
    }

    fn dataset_to_pqcodes(&self, dataset: ArrayView2::<f64>) -> Array2::<usize> {

        let mut pqcodes = Array::from_elem((self.m, self.k), 0)
        // let mut pqcodes = Array::from_elem((dataset.nrows(), self.m), 0);
        for n in 0..dataset.nrows() {
            let codes = vec::new();
            let qpd QuantizedDataPoint {index: n, Vec::with_capacity(capacity: usize)()};
            for m in 0..self.m {
                let begin = self.sub_dimension * m;
                let end = begin + self.sub_dimension - 1;
                let partial_data = dataset.slice(s![n, begin..end]);

                let mut best_centroid: Option::<usize> = None;
                let mut best_distance = f64::NEG_INFINITY;

                for k in 0..self.k {
                    let centroid = &self.codebook.as_ref().unwrap()[[m,k]];
                    let distance = distance::cosine_similarity(&(centroid).view(), &partial_data);
                    if best_distance < distance {
                        best_centroid = Some(k);
                        best_distance = distance;
                    }
                }
                if best_centroid.is_some() {
                    qpd.codes.push(best_centroid.unwrap())
                } 
            }
            pqcodes[] .insert(k, Centroid{id: k, index: n, pqcodes})
        }
        pqcodes
    }

    fn distance_table(&self, query: &ArrayView1<f64>) -> Vec::<Vec::<f64>> {
        let mut dtable = Vec::<Vec::<f64>>::with_capacity(self.m);
        
        for m in 0..self.m {
            dtable.push(Vec::with_capacity(self.k));
            let begin = self.sub_dimension * m;
            let end = begin + self.sub_dimension - 1;
            let partial_data = &query.slice(s![begin..end]);
            for k in 0..self.k {
                let code = &self.codebook.as_ref().unwrap();
                let sub_centroid = &code[[m,k]];
                
                let dist = distance::cosine_similarity(&sub_centroid.view(), &partial_data);
                dtable[m].push(-dist);
            }
        }

        dtable
    }

    fn adist(&self, dtable: Vec::<Vec::<f64>>) -> Array1::<f64> {
        let n = self.dataset.as_ref().unwrap().nrows();
        let pqcode = self.pqcodes.as_ref().unwrap();
        let mut dists = Array1::<f64>::zeros(n);
        for n in 0..n {
            for m in 0..self.m {
                if n == 97478 {
                    println!("%%%%%%%%%%%%%%% \nadist for {} m: {} pqcode: {} dist {}", n, m, pqcode[[n,m]], dtable[m][pqcode[[n,m]]]);
                }
                dists[n] += dtable[m][pqcode[[n,m]]];
            }
            dists[n] = dists[n]/self.m as f64;
            if n == 97478 {
                println!("%%%%%%%%%%%%%%% \nadist for dists[{}] {}", n, dists[n]);
            }
        }
        dists
    }
    
}



impl AlgorithmImpl for ProductQuantization {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn fit(&mut self, dataset: ArrayView2::<f64>) {
        self.dimension = dataset.slice(s![0,..]).len();
        self.sub_dimension = self.dimension / self.m;
        self.dataset = Some(dataset.to_owned());
        
        // Create random selected train data from dataset
        let train_data = self.random_traindata(dataset, self.training_size);
        if self.verbose_print {
            println!("Traing data created shape: {:?}", train_data.shape());
        }

        // Create codebook, [m,k,[sd]] m-th subspace, k-th codewords, sd-th dimension
        // Compute codebook from training data using k-means.
        self.train_codebook(train_data.view());
        if self.verbose_print {
            println!("Codebook created [m, k, d], shape: {:?}", self.codebook.as_ref().unwrap().shape());
        }

        // println!("CODEBOOK: \n {:?}", self.codebook);

        // Compute PQ Codes
        self.pqcodes = Some(self.dataset_to_pqcodes(dataset));
        if self.verbose_print {
            println!("PQ Codes computed, shape {:?}", self.pqcodes.as_ref().unwrap().shape());
        }

        // println!("PQ CODES: \n {:?}", self.pqcodes);
    }

    fn query(&self, query: &ArrayView1::<f64>, result_count: u32) -> Vec<usize> {

        let dtable = self.distance_table(query);
        // println!("Query with dtable\n {:?}", dtable);
        

        let dists = self.adist(dtable);
        // println!("adists:\n {:?}", dists);

        let lookfor = vec![97478, 262700, 846101, 671078, 232287, 727732, 544474, 1133489, 723915, 660281];
        
        let mut best_candidates = BinaryHeap::new();
        for i in 0..dists.len() {
            if lookfor.contains(&(i as i32)) {
                println!("found {} with value {}", i, -dists[i]);
            }
            best_candidates.push(DataEntry{index: i, distance: -dists[i]});
        }

        let mut best_n_candidates: Vec<usize> = Vec::new();
        for _ in 0..result_count {
            let cand = best_candidates.pop().unwrap();
            println!("Candidate {:?}", cand);
            best_n_candidates.push(cand.index);
        }
        
        best_n_candidates.reverse();
        if self.verbose_print {
            println!("best_n_candidates \n{:?}", best_n_candidates);
        }
        best_n_candidates
    }

}