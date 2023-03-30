import csv
import sys
import os


if __name__ == "__main__":
    output_dir = sys.argv[1]

    with open("resultsfinal.csv", "w") as f:
        writer = csv.writer(f)

        # Loop through every directory in output_dir
        directories = os.listdir(output_dir)
        for directory in directories:
            if not os.path.isdir(output_dir + "/" + directory):
                continue
            if not os.path.isfile(output_dir + "/" + directory + "/accuracies.csv"):
                continue
            with open(output_dir + "/" + directory + "/accuracies.csv", "r") as f:
                reader = csv.reader(f)
                data = list(reader)
                writer.writerows(data)
