#!/bin/bash

echo "REMEMBERED TO USE: conda activate thesis, [y]?"
read ready

if [ $ready = 'y' ]
then
    for (( clusters=1; clusters<=10; clusters++ )) do
        for (( train_size=10000; train_size<=100000; train_size=train_size+10000)) do
            echo "Running pq 10 10 255 $train_size 1024 200 $clusters"
            cargo run --release angular glove-100-angular pq 10 10 255 $train_size 1024 200 $clusters
        done
    done

fi