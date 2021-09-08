#!/bin/bash

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

cargo build --release

if [ $run_type = 't' ]
then
    cargo run --release angular $DATASET bruteforce [10]
    #cargo run --release angular random-xs-20-angular bruteforce [10]

    # cargo run --release angular random-xs-20-angular bruteforce [10]
    # cargo run --release angular $DATASET kmeans [10] [50 100] [[1] [2] [3] [4] [5] [10] [30] [50]]
    # cargo run --release angular $DATASET pq [10] [5 500 250000 256 100] [[1 30] [1 60] [1 120] [1 150] [1 500] [5 30] [5 60] [5 120] [5 150] [10 500] [10 30] [10 60] [10 120] [10 150] [10 500]]


elif [ $run_type = 'f' ]
then

    # cargo run --release angular $DATASET bruteforce [10]
    # cargo run --release angular $DATASET kmeans [10] [2000 100] [[16] [24] [32] [64] [128] [192] [256] [384] [512]]
    # cargo run --release angular $DATASET scann_kmeans [10] [2000 10] [[16] [24] [32] [64] [128] [192] [256] [384] [512]]
    cargo run --release angular $DATASET pq [10] [10 2000 250000 256 100] [[16 3072] [24 3072] [32 4096] [64 8192] [64 12288] [72 16384] [88 24576] [128 24576] [192 32768] [192 49152] [384 73728] [384 131072] [512 131072]]
    cargo run --release angular $DATASET pq [10] [10 2000 250000 256 100] [[16 128] [24 128] [32 192] [64 192] [64 256] [72 256] [88 384] [128 384] [192 384] [192 512] [384 512] [384 768] [512 1024]]

    # cargo run --release angular $DATASET pq [10] [13 500 250000 256 100] [[8 512] [8 1024] [16 1024] [16 2048] [16 3072] [24 3072] [32 4096] [64 8192] [64 12288] [72 16384] [88 24576] [128 24576] [192 32768] [192 49152] [384 73728] [384 131072] [512 131072]]
    # cargo run --release angular $DATASET pq [10] [5 2000 250000 256 100] [[8 512] [8 1024] [16 1024] [16 2048] [16 3072] [24 3072] [32 4096] [64 8192] [64 12288] [72 16384] [88 24576] [128 24576] [192 32768] [192 49152] [384 73728] [384 131072] [512 131072]]
    
    # cargo run --release angular $DATASET scann [10] [10 2000 250000 256 50 0.2] [[1 64] [2 128] [4 256] [4 512] [8 512] [8 1024] [16 1024] [16 2048] [16 3072] [24 3072] [32 4096] [64 8192] [64 12288] [72 16384] [88 24576] [128 24576] [192 32768] [192 49152] [384 73728] [384 131072] [512 131072]]
   
else
    exit 0
fi
sh ./copy_results_to_ann.sh

exit 0