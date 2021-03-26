use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, s};
use crate::algs::*;
use rand::{distributions::Uniform, Rng};
use pq_kmeans::{PQKMeans};

#[derive(Debug, Clone)]
pub struct ProductQuantization {
    name: String,
    metric: String,
    dataset: Option<Array2::<f64>>,
    codebook: Option::<Array2::<Array1::<f64>>>,
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

    fn encode_codebook(&mut self, train_data: ArrayView2::<f64>) {
        // Create codebook, [m,k,[sd]] m-th subspace, k-th codewords, sd-th dimension
        let mut codebook = Array::from_elem((self.m, self.k), Array::zeros(self.sub_dimension));
        for m in 0..self.m {
            let begin = self.sub_dimension * m;
            let end = begin + self.sub_dimension - 1;
            let partial_data = train_data.slice(s![.., begin..end]);
            if self.verbose_print {
                println!("Run k-means for m [{}], sub dim {:?}, first element [{}]", m, partial_data.shape(), partial_data[[0,0]]);
            }
            let mut pq_kmeans = PQKMeans::new(self.k, self.max_iterations);
            let codewords = pq_kmeans.run(partial_data.view());
            for (k, (centroid,_)) in codewords.iter().enumerate() {
                    codebook[[m,k]] = centroid.to_owned();
            }
        }
        self.codebook = Some(codebook);
    }

    fn compute_pqcodes(&self, dataset: ArrayView2::<f64>) -> Array2::<usize> {
        let mut pqcodes = Array::from_elem((self.m, dataset.nrows()), 0);
        for idx in 0..dataset.nrows() {
            for m in 0..self.m {
                let begin = self.sub_dimension * m;
                let end = begin + self.sub_dimension - 1;
                let partial_data = dataset.slice(s![idx, begin..end]);


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
                    pqcodes[[m, idx]] = best_centroid.unwrap();
                } 
            }   
        }
        pqcodes
    }

    fn distance_table(&self, query: Array1<f64>) {
        let mut dtable = Vec::<Vec::<f64>>::new();

        for m in 0..self.m {
            let begin = self.sub_dimension * m;
            let end = begin + self.sub_dimension - 1;
            let partial_data = query.slice(s![begin..end]);

            println!("partial_data: {}", partial_data);
             

            for k in 0..self.k {
                let code = &self.codebook.as_ref().unwrap();
                let sub_centroid = &code[[m,k]];
                dtable[m][k] = distance::cosine_similarity(&sub_centroid.view(), &partial_data);
            }
        }
    }
}



impl AlgorithmImpl for ProductQuantization {

    fn __str__(&self) {
        self.name.to_string();
    }

    fn done(&self) {}

    fn get_memory_usage(&self) {}

    fn batch_query(&self) {}

    fn get_batch_results(&self) {}
    
    fn get_additional(&self) {}

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
        self.encode_codebook(train_data.view());
        if self.verbose_print {
            println!("Codebook created [m, k, d], shape: {:?}", self.codebook.as_ref().unwrap().shape());
        }

        // Compute PQ Codes
        let pqcodes = self.compute_pqcodes(dataset);
        if self.verbose_print {
            println!("PQ Codes computed, shape {:?}", pqcodes.shape());
        }
    }

    fn query(&self, p: &ArrayView1::<f64>, result_count: u32) -> Vec<usize> {
        
        let mut best_n_candidates: Vec<usize> = Vec::new();
        best_n_candidates.reverse();
        if self.verbose_print {
            println!("best_n_candidates \n{:?}", best_n_candidates);
        }
        best_n_candidates
    }

}