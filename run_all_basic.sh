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
        ARGS_PQ="[4 128 2000 255 200]"
        CLUSTERS_To_SEARCH_KM="[1]"
        CLUSTERS_To_SEARCH_PQ="[1]";;
    '1' )
        DATASET="glove-25-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[5 1024 20000 255 200]"
        CLUSTERS_To_SEARCH_KM="[32 64 128 256]"
        CLUSTERS_To_SEARCH_PQ="[32 64 128 256]";;
    '2' )n
        DATASET="glove-50-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[10 255 20000 1024 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '3' )
        DATASET="glove-100-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[10 255 20000 1024 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '4' )
        DATASET="glove-200-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[20 255 20000 1024 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '5' )
        DATASET="nytimes-256-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[32 255 20000 1024 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '6' )
        DATASET="deep-image-96-angular"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[16 255 20000 1024 200]"
        CLUSTERS_To_SEARCH_KM="[1 2 3 5 10 20]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    '7' )
        DATASET="lastfm-64-dot"
        ARGS_KM="[1024 200]"
        ARGS_PQ="[16 255 20000 1024 200]"
        CLUSTERS_To_SEARCH_KM="[[1 1] [2 2] [3 3] [5 5] [10 10] [20 20]]"
        CLUSTERS_To_SEARCH_PQ="[1 2 3 5 10 20]";;
    *) printf "Error: Invalid option"
        exit 1;;
esac

echo "Run on $DATASET use [s]ingle or [m]ultiple or [t]est?"
read run_type

# echo "Remove results folder before generating new? [y]?"
# read cmd_remove

# if [ $cmd_remove = 'y' ]
# then
# rm -r results
# fi

cargo build --release

if [ $run_type = 's' ]
then
    '''
    cargo run --release angular random-xs-20-angular bruteforce [10]
    cargo run --release angular random-xs-20-angular kmeans [10] [1024 200] [1 2 3 5 10 20]
    cargo run --release angular random-xs-20-angular pq [10] [10 255 20000 1024 200] [1 2 3 5 10 20]
    '''
    # cargo run --release $METRIC $DATASET bruteforce $RESULTS
    cargo run --release $METRIC $DATASET kmeans $RESULTS $ARGS_KM $CLUSTERS_To_SEARCH_KM
    # cargo run --release $METRIC $DATASET pq $RESULTS $ARGS_PQ $CLUSTERS_To_SEARCH_PQ

elif [ $run_type = 'm' ]
then
    cargo run --release $METRIC $DATASET bruteforce $RESULTS
    for (( k=4; k<=1024; k=k+k )) do
        cargo run --release $METRIC $DATASET kmeans $RESULTS [$k 200] $CLUSTERS_To_SEARCH_KM
    done

    # cargo run --release $METRIC $DATASET pq $RESULTS [1024 255 5000 10 200] $CLUSTERS_To_SEARCH_PQ
    # cargo run --release $METRIC $DATASET pq $RESULTS [1024 255 10000 10 200] $CLUSTERS_To_SEARCH_PQ
    # cargo run --release $METRIC $DATASET pq $RESULTS [1024 255 20000 10 200] $CLUSTERS_To_SEARCH_PQ
    # cargo run --release $METRIC $DATASET pq $RESULTS [1024 255 50000 10 200] $CLUSTERS_To_SEARCH_PQ
elif [ $run_type = 't' ]
then

    cargo run --release $METRIC $DATASET bruteforce [10]
    # cargo run --release $METRIC $DATASET kmeans [10] [64 200] [1 2 3 4 5 10 20 40 64]
    # cargo run --release $METRIC $DATASET kmeans [10] [128 200] [1 2 3 4 5 10 20 40 100 128]
    # cargo run --release $METRIC $DATASET kmeans [10] [256 200] [1 2 3 4 5 10 20 40 100 200 256]
    # cargo run --release $METRIC $DATASET kmeans [10] [512 200] [1 2 3 4 5 10 20 40 100 200 400 512]
    # cargo run --release $METRIC $DATASET kmeans [10] [1024 200] [1 2 3 4 5 10 20 40 50 100 200 400 800 1024]
    # cargo run --release $METRIC $DATASET kmeans [10] [2048 200] [1 2 3 4 5 10 20 40 50 100 200 400 800 1600 2048]
    cargo run --release $METRIC $DATASET kmeans [10] [100 200] [1 2 4 8 30 40 45 50 55 60 65 75 90 110]
    cargo run --release $METRIC $DATASET pq [10] [10 100 2000 255 200] [[1 30] [2 30] [4 30] [8 30] [30 120] [35 100] [40 80] [45 80] [50 80] [55 95] [60 110] [65 110] [75 110] [90 110] [110 120]]
    # cargo run --release $METRIC $DATASET pq [10] [100 128 2000 255 200] [[1 30] [2 30] [4 30] [8 30] [30 120] [35 100] [40 80] [45 80] [50 80] [55 95] [60 110] [65 110] [75 110] [90 110] [110 120] [130 150] [150 200] [170 200] [200 300] [220 500] [250 500] [310 300] [400 300] [500 500] [800 1000]]
    # cargo run --release $METRIC $DATASET pq [10] [100 256 2000 255 200] [[1 30] [2 30] [4 30] [8 30] [30 120] [35 100] [40 80] [45 80] [50 80] [55 95] [60 110] [65 110] [75 110] [90 110] [110 120] [130 150] [150 200] [170 200] [200 300] [220 500] [250 500] [310 300] [400 300] [500 500] [800 1000]]
    # cargo run --release $METRIC $DATASET pq [10] [100 512 2000 255 200] [[1 30] [2 30] [4 30] [8 30] [30 120] [35 100] [40 80] [45 80] [50 80] [55 95] [60 110] [65 110] [75 110] [90 110] [110 120] [130 150] [150 200] [170 200] [200 300] [220 500] [250 500] [310 300] [400 300] [500 500] [800 1000]]
    # cargo run --release $METRIC $DATASET pq [10] [100 1024 2000 255 200] [[1 30] [2 30] [4 30] [8 30] [30 120] [35 100] [40 80] [45 80] [50 80] [55 95] [60 110] [65 110] [75 110] [90 110] [110 120] [130 150] [150 200] [170 200] [200 300] [220 500] [250 500] [310 300] [400 300] [500 500] [800 1000]]

elif [ $run_type = 'full' ]
then
    # cargo run --release $METRIC $DATASET kmeans [10] [2000 200] [[1] [2] [4] [8] [30] [35] [40] [45] [50] [55] [60] [65] [75] [90] [110] [130] [150] [170] [200] [220] [250] [310] [400] [500] [800]]
    cargo run --release $METRIC $DATASET pq [10] [50 2000 250000 255 200] [[1 30] [2 30] [4 30] [8 30] [30 120] [35 100] [40 80] [45 80] [50 80] [55 95] [60 110] [65 110] [75 110] [90 110] [110 120] [130 150] [150 200] [170 200] [200 300] [220 500] [250 500] [310 300] [400 300] [500 500] [800 1000]]
else
    exit 0
fi
sh ./copy_results_to_ann.sh

exit 0