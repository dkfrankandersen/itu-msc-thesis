pub mod hdf5_attributes_fix;
pub mod hdf5_store_file;

pub fn store_results_and_fix_attributes(results: Vec<(f64, std::vec::Vec<(usize, f64)>)>, attrs: hdf5_store_file::Attributes) {
    let store_result = hdf5_store_file::store_results(results, attrs);
    match store_result {
        Ok(file) => {
            hdf5_attributes_fix::run(file).ok();
        },
        Err(e) => {
                    println!("{}", e);
                }
    }
}