#!/bin/bash

DATASET=""
ARGS_KM=""
ARGS_PQ=""
CLUSTERS_To_SEARCH_KM=""
CLUSTERS_To_SEARCH_PQ=""

echo "Is conda activate thesis used [y]?"
read ready

if [ $ready != 'y' ]
then
    exit 1
fi

printf "Datasets:
        [0] random-xs-20-angular (default)
        [1] glove-25-angular
        [2] glove-50-angular
        [3] glove-100-angular
        [4] glove-200-angular
        [5] nytimes-256-angular
        [6] deep-image-96-angular
        [7] lastfm-64-dot"

echo "Select dataset"
read ds

case $ds in
    '0' )
        DATASET="random-xs-20-angular";;
    '1' )
        DATASET="glove-25-angular";;
    '2' )n
        DATASET="glove-50-angular";;
    '3' )
        DATASET="glove-100-angular";;
    '4' )
        DATASET="glove-200-angular";;
    '5' )
        DATASET="nytimes-256-angular";;
    '6' )
        DATASET="deep-image-96-angular";;
    '7' )
        DATASET="lastfm-64-dot";;
    *) printf "Error: Invalid option"
        exit 1;;
esac

echo "Run on $DATASET use [t]est or [f]ull?"
read run_type

# echo "Remove results folder before generating new? [y]?"
# read cmd_remove

# if [ $cmd_remove = 'y' ]
# then
# rm -r results
# fi

cargo build --release

if [ $run_type = 't' ]
then
    # cargo run --release angular $DATASET bruteforce [10]
    cargo run --release angular $DATASET kmeans [10] [50 100] [[1] [2] [3] [4] [5] [10] [30] [50]]
    cargo run --release angular $DATASET kmeans [10] [500 100] [[1] [2] [3] [4] [5] [10] [30] [60] [120] [150] [500]]
    cargo run --release angular $DATASET kmeans [10] [1000 100] [[1] [2] [3] [4] [5] [10] [30] [60] [120] [150] [500]]
    cargo run --release angular $DATASET kmeans [10] [2000 100] [[1] [2] [3] [4] [5] [10] [30] [60] [120] [150] [500]]

    cargo run --release angular $DATASET pq [10] [5 500 250000 256 100] [[1 30] [1 60] [1 120] [1 150] [1 500] [5 30] [5 60] [5 120] [5 150] [10 500] [10 30] [10 60] [10 120] [10 150] [10 500]]
    cargo run --release angular $DATASET pq [10] [5 500 250000 256 100] [[1 30] [1 60] [1 120] [1 150] [1 500] [5 30] [5 60] [5 120] [5 150] [10 500] [10 30] [10 60] [10 120] [10 150] [10 500]]
    cargo run --release angular $DATASET pq [10] [5 1000 250000 256 100] [[1 30] [1 60] [1 120] [1 150] [1 500] [5 30] [5 60] [5 120] [5 150] [10 500] [10 30] [10 60] [10 120] [10 150] [10 500]]
    cargo run --release angular $DATASET pq [10] [5 1000 250000 256 100] [[1 30] [1 60] [1 120] [1 150] [1 500] [5 30] [5 60] [5 120] [5 150] [10 500] [10 30] [10 60] [10 120] [10 150] [10 500]]
    cargo run --release angular $DATASET pq [10] [5 2000 250000 256 100] [[1 30] [1 60] [1 120] [1 150] [1 500] [5 30] [5 60] [5 120] [5 150] [10 500] [10 30] [10 60] [10 120] [10 150] [10 500]]
    # cargo run --release angular $DATASET pq [10] [5 2000 250000 256 100] [[2 30] [2 60] [2 120] [1 150] [2 500]]
    # cargo run --release angular $DATASET pq [10] [5 2000 250000 256 100] [[10 30] [10 60] [10 120] [10 150] [10 500]]
    # cargo run --release angular $DATASET pq [10] [5 2000 250000 256 100] [[100 30] [100 60] [100 120] [100 150] [100 500]]

    # cargo run --release angular $DATASET pq [10] [10 2000 250000 256 200] [[1 30] [1 60] [1 120] [1 150] [1 500]]
    # cargo run --release angular $DATASET pq [10] [10 2000 250000 256 100] [[2 30] [2 60] [2 120] [1 150] [2 500]]
    # cargo run --release angular $DATASET pq [10] [10 2000 250000 256 100] [[10 30] [10 60] [10 120] [10 150] [10 500]]
    # cargo run --release angular $DATASET pq [10] [10 2000 250000 256 100] [[100 30] [100 60] [100 120] [100 150] [100 500]]

    # cargo run --release angular $DATASET pq [10] [20 2000 250000 256 200] [[1 30] [1 60] [1 120] [1 150] [1 500]]
    # cargo run --release angular $DATASET pq [10] [20 2000 250000 256 100] [[2 30] [2 60] [2 120] [1 150] [2 500]]
    # cargo run --release angular $DATASET pq [10] [20 2000 250000 256 100] [[10 30] [10 60] [10 120] [10 150] [10 500]]
    # cargo run --release angular $DATASET pq [10] [20 2000 250000 256 100] [[100 30] [100 60] [100 120] [100 150] [100 500]]

    # cargo run --release angular $DATASET pq [10] [50 2000 250000 256 200] [[1 30] [1 60] [1 120] [1 150] [1 500]]
    # cargo run --release angular $DATASET pq [10] [20 2000 250000 256 100] [[2 30] [2 60] [2 120] [1 150] [2 500]]
    # cargo run --release angular $DATASET pq [10] [20 2000 250000 256 100] [[10 30] [10 60] [10 120] [10 150] [10 500]]
    # cargo run --release angular $DATASET pq [10] [20 2000 250000 256 100] [[100 30] [100 60] [100 120] [100 150] [100 500]]

elif [ $run_type = 'f' ]
then
    # cargo run --release angular $DATASET kmeans [10] [2000 100] [[1] [5] [10] [20] [50] [100] [200] [500]]
    # cargo run --release angular $DATASET pq [10] [4 2000 250000 256 100] [[1 100] [30 300] [20 30] [20 500] [20 700] [30 1400]]
    # cargo run --release angular $DATASET pq [10] [5 2000 250000 256 100] [[1 100] [30 300] [20 30] [20 500] [20 700] [30 1400]]
    cargo run --release angular $DATASET pq [10] [10 2000 250000 256 100] [[1 100] [30 300] [20 30] [20 500] [20 700] [30 1400]]
    # cargo run --release angular $DATASET pq [10] [25 2000 250000 256 100] [[1 100] [30 300] [20 30] [20 500] [20 700] [30 1400]]
    # cargo run --release angular $DATASET pq [10] [50 2000 250000 256 100] [[1 100] [30 300] [20 30] [20 500] [20 700] [30 1400]]

else
    exit 0
fi
sh ./copy_results_to_ann.sh

exit 0