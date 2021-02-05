// mod hdf5_example2;
mod datasets;

fn main() {
    println!("Hello, world!");

    // let res = hdf5_example2::run();
    // println!("{:?}", res);

    let _res = datasets::get_data_array("../../datasets/glove-100-angular.hdf5", "train").unwrap();
    for x in _res.iter() {
        println!("{}", x);
    }
    
    let _res2 = datasets::as_2d_array("../../datasets/glove-100-angular.hdf5", "test");
    let _res3 = datasets::as_2d_array("../../datasets/glove-100-angular.hdf5", "neighbors");
    let _res4 = datasets::as_2d_array("../../datasets/glove-100-angular.hdf5", "distances");
    
}
