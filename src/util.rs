pub fn store_results() {
    let path: &str = "results/";
    let filename: &str = "test.hdf5";

    let full_path = format!("{}{}", path, filename);

    println!("{}", full_path);
    let file = hdf5::File::create(full_path);
}

