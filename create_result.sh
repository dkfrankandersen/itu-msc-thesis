#!/bin/bash

echo "REMEMBERED TO USE: conda activate thesis, [y]?"
read ready

if [ $ready = 'y' ]
then

    echo "Building with cargo, ignore dead code"
    RUSTFLAGS="$RUSTFLAGS -A dead_code" cargo build

    for (( clusters=1; clusters<=1000; clusters=clusters*2 )) do
    echo "Running kmeans 10 $clusters 200 1"
    cargo run --release cosine glove-100-angular kmeans 10 $clusters 200 1
    done

fi