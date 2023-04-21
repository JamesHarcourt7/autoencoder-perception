import csv
import sys
import os


if __name__ == "__main__":
    output_dir = sys.argv[1]

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

        new_data = []
        decay = directory.split("_")[0][-3:]
        decay = list(decay)
        decay.insert(1, '.')
        decay = float("".join(decay))

        for i in range(len(data)):
            if (data[i]):
                if data[i][0] == "Data Indexes":
                    new_data.append(data[i])
                    new_data.append(["Decay", decay])
                else:
                    new_data.append(data[i])

        with open(output_dir + "/" + directory + "/accuracies.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(new_data)
