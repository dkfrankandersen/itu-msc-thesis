#!/bin/bash

METRIC="angular"

DATASET="random-xs-20-angular"
# DATASET="glove-100-angular"
# DATASET="nytimes-256-angular"
# DATASET="deep-image-96-angular"
# DATASET="lastfm-64-dot"
# DATASET="glove-25-angular"
# DATASET="glove-50-angular"
# DATASET="glove-200-angular"
RESULTS="[10]"
CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]";
CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";


echo "Is conda activate thesis used [y]?"
read ready

if [ $ready != 'y' ]
then
    exit 1
fi

echo "Run on $DATASET use [s]ingle or [m]ultiple?"
read run_type

cargo build --release

if [ $run_type = 's' ]
then
    cargo run --release $METRIC $DATASET bruteforce $RESULTS
    cargo run --release $METRIC $DATASET kmeans $RESULTS [1024 200] $CLUSTERS_To_SEARCH_KM
    cargo run --release $METRIC $DATASET pq $RESULTS [1024 255 5000 10 200] $CLUSTERS_To_SEARCH_PQ

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

exit 0