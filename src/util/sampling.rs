
use rand::{prelude::*};

pub fn sampling_without_replacement<T: RngCore>(mut rng: T, max: usize, n: usize) -> Vec<usize> {
    // let mut rng = StdRng::seed_from_u64(42);
    let sample: Vec<usize> = (0..max).collect();
    let mut choosen_indexes:Vec<&usize> = sample.iter().choose_multiple(&mut rng, n);
    choosen_indexes.shuffle(&mut rng);
    let mut unique_indexes = Vec::<usize>::new();
    for e in choosen_indexes {
        unique_indexes.push(*e);
    }
    unique_indexes
}