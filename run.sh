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

# cargo build --release

if [ $run_type = 't' ]
then
    # cargo run --release angular $DATASET bruteforce _ [10]
    # cargo run --release angular $DATASET kmeans _K2000 [10] [2000 10] [[5] [10] [15] [25] [50] [100] [200] [250] [500]]
    # cargo run --release angular $DATASET pq _M20_C16_TR25 [10] [20 2000 250000 16 10] [[250 2000] [300 1000] [300 2000] [350 1000] [350 1500]]
    # cargo run --release angular $DATASET pq _M25_C16_TR25 [10] [25 2000 250000 16 10] [[250 2000] [300 1000] [300 2000] [350 1000] [350 1500]]
    
    cargo run --release angular $DATASET scann _M20_C16_T02_TR25 [10] [20 2000 250000 16 10 0.2] [[125 20000] [150 30000] [200 30000]]
    cargo run --release angular $DATASET scann _M25_C16_T02_TR25 [10] [25 2000 250000 16 10 0.2] [[125 20000] [150 30000] [200 30000]]
    cargo run --release angular $DATASET scann _M10_C256_T02_TR25 [10] [10 2000 250000 256 10 0.2] [[125 20000] [150 30000] [200 30000]]
    cargo run --release angular $DATASET scann _M50_C16_T02_TR25 [10] [50 2000 250000 16 10 0.2] [[125 20000] [150 30000] [200 30000]]

elif [ $run_type = 'b' ]
then
    cargo run --release angular $DATASET bruteforce _ [10]
elif [ $run_type = 'k' ]
then
    cargo run --release angular $DATASET kmeans _ [10] [2000 10] [[16] [24] [32] [64] [128] [192] [256] [384] [512]]
elif [ $run_type = 'p' ]
then
    cargo run --release angular $DATASET pq _ [10] [50 2000 1183514 16 10] [[16 512] [24 1024] [64 2048] [72 3072] [88 4096] [128 8192] [192 16384] [256 32768] [384 73728]]
elif [ $run_type = 's' ]
then
    cargo run --release angular $DATASET scann _ [10] [50 2000 1183514 16 10 0.2] [[1 30] [2 30] [4 30] [8 30] [30 120] [35 100] [40 80] [45 80] [50 80] [55 95] [60 110] [65 110] [75 110] [90 110] [110 120] [130 150] [150 200] [170 200] [200 300] [220 500] [250 500] [310 300] [400 300] [500 500] [800 1000]]
else
    exit 0
fi

# sh ./copy_results.sh

exit 0