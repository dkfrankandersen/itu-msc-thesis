#!/bin/bash

echo "REMEMBERED TO USE: conda activate thesis, [y]?"
read ready

if [ $ready = 'y' ]
then
    for (( k=10; k<=1300; k=k+k )) do
        for (( s=1; s<=10; s++ )) do
            echo "Running kmeans 10 $k 200 $s"
            cargo run --release angular glove-100-angular kmeans 10 $k 200 $s
        done
    done

fi