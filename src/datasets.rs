
pub fn get_dataset() -> hdf5::Result<()> {
    // so that libhdf5 doesn't print errors to stdout
    let _e = hdf5::silence_errors();
    {
        let file = hdf5::File::open("../../datasets/glove-100-angular.hdf5")?;

        
        println!("HDF5 member_names: {:?}", file.member_names()?);
        for x in file.member_names()? {
            println!("{:?}", x)
        }
        
        // println!("HDF5 size: {:?}", file.size());
        // println!("HDF5 access_plist: {:?}", file.access_plist());
        // println!("HDF5 create_plist: {:?}", file.create_plist());

        let train_dataset = file.dataset("train")?;
        let train:ndarray::Array2::<f32> = train_dataset.read_2d::<f32>()?;
        println!("train_dataset shape {:?}", train_dataset.shape());
        println!("Vector length {:?}", train_dataset.shape()[1]);
        println!("{:?}", train);
        println!("Array length {:?}", train.shape()[0]);
        println!("{}", train[[0,0]]);
        // for i in 0..train.shape()[0] {
        //     println!("{}: {:?}", i, train[[i,0]]);
        // }
            
        // for x in train.genrows() {
        //     println!("{:?}", x);
        // }

        // for (i, row) in train.iter().enumerate() {
        //     for (y, col) in train.iter().enumerate() {
        //         println!("{}", y);
        //     }
        // }

        // let test_queries = file.dataset("test")?;
        // println!("test_queries shape {:?}", test_queries.shape());
        // println!("{:?}", test_queries.read_2d::<f32>()?);

        // let neighbors = file.dataset("neighbors")?;
        // println!("neighbors shape {:?}", neighbors.shape());
        // println!("{:?}", neighbors.read_2d::<i32>()?);

        // let distance = file.dataset("distance")?;
        // println!("distance shape {:?}", distance.shape());
        // println!("{:?}", distance.read_2d::<f32>()?);
    }

    Ok(())
}