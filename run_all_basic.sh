#!/bin/bash

METRIC="angular"
RESULTS="[10]"

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
        DATASET="random-xs-20-angular"
        ARGS_KM="[128 200]"
        ARGS_PQ="[128 255 2000 10 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '1' )
        DATASET="glove-25-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[1024 255 20000 5 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '2' )
        DATASET="glove-50-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[1024 255 20000 10 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '3' )
        DATASET="glove-100-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[1024 255 20000 10 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '4' )
        DATASET="glove-200-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[1024 255 20000 10 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '5' )
        DATASET="nytimes-256-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[1024 255 20000 10 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '6' )
        DATASET="deep-image-96-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[1024 255 20000 10 200]";
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '7' )
        DATASET="lastfm-64-dot"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[1024 255 20000 10 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    *) printf "Error: Invalid option"
        exit 1;;
esac

echo "Run on $DATASET use [s]ingle or [m]ultiple?"
read run_type

cargo build --release

if [ $run_type = 's' ]
then
    cargo run --release $METRIC $DATASET bruteforce $RESULTS
    cargo run --release $METRIC $DATASET kmeans $RESULTS $ARGS_KM $CLUSTERS_To_SEARCH_KM
    cargo run --release $METRIC $DATASET pq $RESULTS $ARGS_PQ $CLUSTERS_To_SEARCH_PQ
fi

if [ $run_type = 'm' ]
then
    cargo run --release $METRIC $DATASET bruteforce $RESULTS
    for (( k=4; k<=1024; k=k+k )) do
        cargo run --release $METRIC $DATASET kmeans $RESULTS [$k 200] $CLUSTERS_To_SEARCH_KM
    done

    cargo run --release $METRIC $DATASET pq $RESULTS [1024 255 5000 10 200] $CLUSTERS_To_SEARCH_PQ
    cargo run --release $METRIC $DATASET pq $RESULTS [1024 255 10000 10 200] $CLUSTERS_To_SEARCH_PQ
    cargo run --release $METRIC $DATASET pq $RESULTS [1024 255 20000 10 200] $CLUSTERS_To_SEARCH_PQ
    cargo run --release $METRIC $DATASET pq $RESULTS [1024 255 50000 10 200] $CLUSTERS_To_SEARCH_PQ
fi

sh ./copy_results_to_ann.sh

exit 0