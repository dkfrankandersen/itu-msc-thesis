// pub mod cosine_similarity;
// pub mod euclidian;
// use crate::distances::cosine_similarity::CosineSimilarity;
// use crate::distances::euclidian::Euclidian;

// pub enum Distance {
//     CosineSimilarity(CosineSimilarity),
//     Euclidian(Euclidian)
// }

// pub fn distance_factory(metric: &str) -> Distance {
//     match metric {
//         "cosine_similarity" => Distance::CosineSimilarity(CosineSimilarity::new()),
//         "euclidian" => Distance::Euclidian(Euclidian::new()),
//         _ => panic!("Unknown distance metric")
//     }
// }

// pub fn get_dist() -> f64 {
//     match distance_factory("cosine_similarity") {
//         Distance::CosineSimilarity( obj ) => CosineSimilarity::dist(),
//         Distance::Euclidian( obj ) => Euclidian::dist()
//     }
// }
