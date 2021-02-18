use ndarray::{Array2, ArrayView1, Array1};

pub fn get_dataset(filename: &str) -> hdf5::File {
    return hdf5::File::open(filename).unwrap();
}


pub fn get_dataset_f64(file: &hdf5::File, dataset: &str) -> Array2::<f64> {
    return (file).dataset(dataset).unwrap().read_2d::<f64>().unwrap();
}

pub fn get_dataset_i64(file: &hdf5::File, dataset: &str) -> Array2::<usize> {
    return (file).dataset(dataset).unwrap().read_2d::<usize>().unwrap();
}

pub fn get_dataset_usize(file: &hdf5::File, dataset: &str) -> Array2::<usize> {
    return (file).dataset(dataset).unwrap().read_2d::<usize>().unwrap();
}

pub fn normalize_vector(p: &ArrayView1::<f64>)-> Array1::<f64>  {
    let magnitude = p.dot(p).sqrt();
    return p.map(|e| e/magnitude);
}

pub fn normalize_all(dataset: Array2<f64>) -> Array2::<f64> {
    let mut ds_new: Array2<f64> = dataset.clone();
    for (idx_row, p) in dataset.outer_iter().enumerate() {
        let magnitude = p.dot(&p).sqrt();        
        for (idx_col, val) in p.iter().enumerate() {
            ds_new[[idx_row, idx_col]] = val/magnitude;
        }            
    }
    return ds_new;
}