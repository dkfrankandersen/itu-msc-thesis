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

echo "Run on $DATASET use [t]est, [b]ruteforce, [k]means, [p]roduct quantization or [s]cann?"
read run_type

cargo build --release

if [ $run_type = 't' ]
then
    # cargo run --release angular $DATASET kmeans [10] [2000 10] [[16] [24] [32] [64] [128] [192] [256] [384] [512]]
    # cargo run --release angular $DATASET pq [10] [10 2000 250000 256 10] [[16 512] [24 1024] [64 2048] [72 3072] [88 4096] [128 8192] [128 12288] [192 16384] [256 32768] [384 73728]]
    cargo run --release angular $DATASET scann [10] [10 2000 250000 256 10 0.2] [[1 32]]
elif [ $run_type = 'b' ]
then
    cargo run --release angular $DATASET bruteforce [10]
elif [ $run_type = 'k' ]
then
    cargo run --release angular $DATASET kmeans [10] [2000 10] [[16] [24] [32] [64] [128] [192] [256] [384] [512]]
elif [ $run_type = 'p' ]
then
    cargo run --release angular $DATASET pq [10] [50 2000 250000 16 10] [[16 512] [24 1024] [64 2048] [72 3072] [88 4096] [128 8192] [128 12288] [192 16384] [256 32768] [384 73728]]
elif [ $run_type = 's' ]
then
    cargo run --release angular $DATASET scann [10] [50 2000 1183514 16 1 0.2] [[16 512] [24 1024] [64 2048] [72 3072] [88 4096] [128 8192] [128 12288] [192 16384] [256 32768] [384 73728]]
    # cargo run --release angular $DATASET scann [10] [50 2000 250000 16 10 0.2] [[1 30] [2 30] [4 30] [8 30] [30 120] [35 100] [40 80] [45 80] [50 80] [55 95] [60 110] [65 110] [75 110] [90 110] [110 120] [130 150] [150 200] [170 200] [200 300] [220 500] [250 500] [310 300] [400 300] [500 500] [800 1000]]
else
    exit 0
fi

sh ./copy_results.sh

exit 0