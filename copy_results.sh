#!/bin/bash
# echo "Remove results in ann benchmark folder [y]?"
# read cmd_remove

# if [ $cmd_remove = 'y' ]
# then
# rm -r /d/repository/itu/thesis/ann-benchmarks/results
# fi

echo "Copy results in ann benchmark folder [y]?"
read cmd_copy
if [ $cmd_copy = 'y' ]
then
cp -r results ../ann-benchmarks/
fi

exit 0