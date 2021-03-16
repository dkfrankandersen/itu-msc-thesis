#!/bin/bash

echo "Changing conda enviorment to thesis"
conda activate thesis

echo "Building with cargo"
cargo build

for (( clusters=1; clusters<1000; clusters=clusters+10; )) do
echo "Running kmeans 10 $clusters 200 1"
cargo run --release cosine glove-100-angular kmeans 10 $clusters 200 1
done