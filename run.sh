#!/bin/bash

model=("none" "no-mask" "mask" "dims")
count=0

for m in "${model[@]}"; do
    for i in {1..50}; do
        output_dir="output_${count}"
        ((count++))
        mkdir "out"
        cd "out"
        mkdir "$output_dir"
        cd ..


        # Run the Python script with the current parameter combination and output directory
        python environment.py novis "$m" "out/$output_dir"
    done
done

# Aggregate results in 1 csv file
python aggregate.py "out"
      