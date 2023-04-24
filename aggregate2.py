import csv
import sys
import os
import numpy as np

if __name__ == "__main__":
    output_dir = sys.argv[1]

    with open("decisions_final.csv", "w") as f:
        writer = csv.writer(f)

        # Loop through every directory in output_dir
        directories = os.listdir(output_dir)
        for directory in directories:
            if not os.path.isdir(output_dir + "/" + directory):
                continue
            if not os.path.isfile(output_dir + "/" + directory + "/accuracies.csv"):
                continue
            with open(output_dir + "/" + directory + "/decisions.csv", "r") as f:
                reader = csv.reader(f)
                data = list(reader)
                if type(data[0]) == int:
                    writer.writerow(['n', 1])
                else:
                    writer.writerow(['n', len(data[0])])
                writer.writerows(data)
