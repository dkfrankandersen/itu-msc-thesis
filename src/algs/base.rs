
struct Algorithm() {}

trait AlgorithmImpl {
    fn done(&self);

    fn get_memory_usage(&self);

    fn fit(&self);

    fn query(&self);

    fn batch_query(&self);

    fn get_batch_results(&self);
    
    fn get_additional(&self);

    fn __str__(&self);
}