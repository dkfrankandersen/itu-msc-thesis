
pub fn get_dataset() -> hdf5::Result<()> {

    // so that libhdf5 doesn't print errors to stdout
    let _e = hdf5::silence_errors();
    
    {
        let file = hdf5::File::open("../../datasets/glove-100-angular.hdf5")?;

        
        println!("HDF5 member_names: {:?}", file.member_names()?);
        for x in file.member_names()? {
            println!("{:?}", x)
        }
        
        println!("HDF5 size: {:?}", file.size());
        println!("");
        println!("HDF5 access_plist: {:?}", file.access_plist());
        println!("");
        println!("HDF5 create_plist: {:?}", file.create_plist());
        println!("");

        let train_dataset = file.dataset("train")?;
        println!("train_dataset shape {:?}", train_dataset.shape());
        println!("Vector length {:?}", train_dataset.shape()[1]);
        // println!("Shape {:?}", train_dataset.shape());
        // println!("{:?}", train_dataset.read_raw::<f32>());
        println!("{:?}", train_dataset.read_2d::<f32>()?);

        let test_queries = file.dataset("test")?;
        println!("test_queries shape {:?}", test_queries.shape());
        println!("{:?}", test_queries.read_2d::<f32>()?);

        let neighbors = file.dataset("neighbors")?;
        println!("neighbors shape {:?}", neighbors.shape());
        println!("{:?}", neighbors.read_2d::<i32>()?);

        let distance = file.dataset("distance")?;
        println!("distance shape {:?}", distance.shape());
        println!("{:?}", distance.read_2d::<f32>()?);
    }

    Ok(())
}