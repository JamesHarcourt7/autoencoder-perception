#!/bin/bash

model=("none")
count=0

for m in "${model[@]}"; do
    for i in {1..50}; do
        output_dir="output_${count}"
        ((count++))
        mkdir "logs/$output_dir"

        # Run the Python script with the current parameter combination and output directory
        python environment2.py novis 4 "$m" "logs/$output_dir"
    done
done

# Aggregate results in 1 csv file
python aggregate.py "logs"
      