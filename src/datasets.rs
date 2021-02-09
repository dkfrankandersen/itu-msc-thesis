
pub fn get_data_array(filename: &str, dataset: &str) -> hdf5::Result<ndarray::Array2::<f32>> {
    // so that libhdf5 doesn't print errors to stdout
    let _e = hdf5::silence_errors();
    let file = hdf5::File::open(filename)?;

    println!("Loading: {} from {}", dataset, filename);
    println!("Datasets in file: {:?}", file.member_names()?);
    
    let hdf5_data = file.dataset(dataset)?;
    let mut res = ndarray::Array2::<f32>::zeros((0,0));
    if dataset == "train" || dataset == "test" || dataset == "distances"  {
        let data_2d_float:ndarray::Array2::<f32> = hdf5_data.read_2d::<f32>()?;
        res = data_2d_float;
        println!("dataset shape: {:?}", res.shape());
    } else if dataset == "neighbors" {
        let data_2d_int:ndarray::Array2::<i32> = hdf5_data.read_2d::<i32>()?;
        println!("dataset shape: {:?}", data_2d_int.shape());
    } else {
        println!("ERROR: WRONG DATASET CHOOSEN");
    }
    Ok(res)
}

pub fn as_2d_array(filename: &str, dataset: &str) -> ndarray::Array2::<f32> {
    match get_data_array(filename, dataset) {
        Ok(_v) => return _v,
        Err(_e) => return ndarray::Array2::<f32>::zeros((0,0))
    }
}