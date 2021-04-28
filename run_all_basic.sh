#!/bin/bash

METRIC="angular"
DATASET="random-xs-20-angular"

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
    cargo run --release $METRIC $DATASET bruteforce 10
    cargo run --release $METRIC $DATASET kmeans 10 8 200 3
    cargo run --release $METRIC $DATASET pq 10 8 255 1000 10 200 3
fi

if [ $run_type = 'm' ]
then
    cargo run --release $METRIC $DATASET bruteforce 10
    for (( k=4; k<=2048; k=k*2 )) do
        for (( r=1; r<=$k/4; r++ )) do
            cargo run --release $METRIC $DATASET kmeans 10 $k 200 $r
            cargo run --release $METRIC $DATASET pq 10 $k 255 1000 10 200 $r
        done
    done
fi

exit 0