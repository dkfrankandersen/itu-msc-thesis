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
    '2' )
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
    cargo run --release angular $DATASET bruteforce [10]
    cargo run --release angular random-xs-20-angular bruteforce [10]
    # cargo run --release angular random-xs-20-angular bruteforce [10]
    # cargo run --release angular $DATASET kmeans [10] [50 100] [[1] [2] [3] [4] [5] [10] [30] [50]]
    # cargo run --release angular $DATASET pq [10] [5 500 250000 256 100] [[1 30] [1 60] [1 120] [1 150] [1 500] [5 30] [5 60] [5 120] [5 150] [10 500] [10 30] [10 60] [10 120] [10 150] [10 500]]


elif [ $run_type = 'f' ]
then

    # cargo run --release angular $DATASET kmeans [10] [2000 100] [[8] [16] [24] [32] [64] [128] [192] [256] [384] [512]]
    cargo run --release angular $DATASET pq [10] [10 2000 250000 256 100] [[8 512] [8 1024] [16 1024] [16 2048] [16 3072] [24 3072] [32 4096] [64 8192] [64 12288] [72 16384] [88 24576] [128 24576] [192 32768] [192 49152] [384 73728] [384 131072] [512 131072]]
    # cargo run --release angular $DATASET pq [10] [13 500 250000 256 100] [[8 512] [8 1024] [16 1024] [16 2048] [16 3072] [24 3072] [32 4096] [64 8192] [64 12288] [72 16384] [88 24576] [128 24576] [192 32768] [192 49152] [384 73728] [384 131072] [512 131072]]
    # cargo run --release angular $DATASET pq [10] [5 2000 250000 256 100] [[8 512] [8 1024] [16 1024] [16 2048] [16 3072] [24 3072] [32 4096] [64 8192] [64 12288] [72 16384] [88 24576] [128 24576] [192 32768] [192 49152] [384 73728] [384 131072] [512 131072]]
    
    # cargo run --release angular $DATASET scann [10] [10 2000 250000 256 50 0.2] [[1 64] [2 128] [4 256] [4 512] [8 512] [8 1024] [16 1024] [16 2048] [16 3072] [24 3072] [32 4096] [64 8192] [64 12288] [72 16384] [88 24576] [128 24576] [192 32768] [192 49152] [384 73728] [384 131072] [512 131072]]

    # cargo run --release angular $DATASET pq [10] [10 2000 250000 256 100] [[1 16] [1 32] [1 64] [1 128] [1 256] [1 512] [2 16] [2 32] [2 64] [2 128] [2 256] [2 512] [4 16] [4 32] [4 64] [4 128] [4 256] [4 512] [8 16] [8 32] [8 64] [8 128] [8 256] [8 512] [16 16] [16 32] [16 64] [16 128] [16 256] [16 512] [32 16] [32 32] [32 64] [32 128] [32 256] [32 512] [64 16] [64 32] [64 64] [64 128] [64 256] [64 512] [128 16] [128 32] [128 64] [128 128] [128 256] [128 512] [256 16] [256 32] [256 64] [256 128] [256 256] [256 512] [512 16] [512 32] [512 64] [512 128] [512 256] [512 512] [1024 16] [1024 32] [1024 64] [1024 128] [1024 256] [1024 512] [2048 16] [2048 32] [2048 64] [2048 128] [2048 256] [2048 512] ]
    
else
    exit 0
fi
sh ./copy_results_to_ann.sh

exit 0