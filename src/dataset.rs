use ndarray::{Array2, ArrayBase, ViewRepr, Dim, OwnedRepr};

pub fn get_dataset(filename: &str) -> hdf5::File {
    return hdf5::File::open(filename).unwrap();
}


pub fn get_dataset_f32(file: &hdf5::File, dataset: &str) -> Array2::<f32> {
    return (file).dataset(dataset).unwrap().read_2d::<f32>().unwrap();
}

pub fn get_dataset_i32(file: &hdf5::File, dataset: &str) -> Array2::<i32> {
    return (file).dataset(dataset).unwrap().read_2d::<i32>().unwrap();
}

pub fn normalize_vector(p: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>) 
                        -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>> {
    let magnitude = p.dot(p).sqrt();
    return p.map(|e| e/magnitude);
}

pub fn normalize_all(dataset: Array2<f32>) -> Array2::<f32> {
    let mut ds_new: Array2<f32> = dataset.clone();
    for (idx_row, p) in dataset.outer_iter().enumerate() {
        let magnitude = p.dot(&p).sqrt();        
        for (idx_col, val) in p.iter().enumerate() {
            ds_new[[idx_row, idx_col]] = val/magnitude;
        }            
    }
    return ds_new;
}