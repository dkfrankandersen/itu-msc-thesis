// #[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
// #[repr(u8)]
// pub enum Color {
//     RED = 1,
//     GREEN = 2,
//     BLUE = 3,
// }

// #[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
// #[repr(C)]
// pub struct Pixel {
//     xy: (f32, i64),
//     color: Color,
// }

pub fn run() -> hdf5::Result<()> {
    use ndarray::arr1;

    // so that libhdf5 doesn't print errors to stdout
    let _e = hdf5::silence_errors();
    {
        // write
        let file = hdf5::File::create("../../datasets/Vec100.h5")?;
        let ds_distance = file.new_dataset::<f32>().create("distance", (2,6))?;
        let _ds_neighbors = file.new_dataset::<i32>().create("neighbors", (2,100))?;
        let _ds_test = file.new_dataset::<f32>().create("test", (2,100))?;
        let _ds_train = file.new_dataset::<f32>().create("train", (2,100))?;
        
        ds_distance.write(& arr1(&
                [[0.430391, 0.430392, 0.43033, 0.430392, 0.43033, 0.43033],
                [0.430391, 0.430392, 0.43033, 0.430392, 0.43033, 0.43033]],
            ))?;
    }
    // {
    //     // read
    //     let file = hdf5::File::open("../../datasets/pixels.h5")?;
    //     let colors = file.dataset("colors")?;
    //     assert_eq!(colors.read_1d::<Color>()?, arr1(&[RED, BLUE]));
    //     let pixels = file.dataset("dir/pixels")?;
    //     assert_eq!(
    //         pixels.read_raw::<Pixel>()?,
    //         vec![
    //             Pixel { xy: (1, 2), color: RED },
    //             Pixel { xy: (3, 4), color: BLUE },
    //             Pixel { xy: (5, 6), color: GREEN },
    //             Pixel { xy: (7, 8), color: RED },
    //         ]
    //     );
    // }
    Ok(())
}