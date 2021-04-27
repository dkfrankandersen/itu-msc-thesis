#!/bin/bash

echo "REMEMBERED TO USE: conda activate thesis, [y]?"
read ready

if [ $ready = 'y' ]
then
    cargo run --release angular glove-25-angular bruteforce 10
    cargo run --release angular glove-25-angular kmeans 10 255 200 3
    cargo run --release angular glove-25-angular pq 10 10 255 10000 1024 200 3
fi