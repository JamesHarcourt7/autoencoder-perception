import csv
import matplotlib.pyplot as plt
import numpy as np

model_lables = {"none": "Baseline", "no-mask": "1P AE No Mask", "mask": "Autoencoder", "dims": "1P AE Increased Latent Space", "split": "2P AE With Mask", "split-dims": "2P AE Increased Latent Space", "vae-nosplit": "1P VAE", "vae-split": "2P VAE"}

class Entry:

    def __init__(self, digit, model, mse, psnr, ssim, explorations):
        self.digit = digit
        self.model = model
        self.mse = mse
        self.psnr = psnr
        self.ssim = ssim
        self.explorations = explorations


with open('resultsfinal.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

    entries = list()

    for i in range(len(data)):
        if (data[i]):
            if data[i][0] == "Data Index":
                digit = data[i][1]
                model = data[i+1][1]
                mse = np.array(data[i+2][1:]).astype(np.float32)
                psnr = np.array(data[i+3][1:]).astype(np.float32)
                ssim = np.array(data[i+4][1:]).astype(np.float32)
                explorations = np.array(data[i+5][1:]).astype(np.float32) * 100
                entry = Entry(digit, model, mse, psnr, ssim, explorations)
                entries.append(entry)

    # Sort entries by model in same order as model_lables keys
    entries = sorted(entries, key=lambda entry: list(model_lables.keys()).index(entry.model))

max_mse = np.max([np.max(entry.mse) for entry in entries])
min_mse = 0
max_psnr = np.max([np.max(entry.psnr) for entry in entries])
min_psnr = np.min([np.min(entry.psnr) for entry in entries])
max_ssim = np.max([np.max(entry.ssim) for entry in entries])
min_ssim = np.min([np.min(entry.ssim) for entry in entries])

# Group entries by digit
digits = dict()
for entry in entries:
    if entry.digit not in digits:
        digits[entry.digit] = list()
    digits[entry.digit].append(entry)

# Group by model
models = dict()
for digit in digits:
    if digit not in models:
        models[digit] = dict()
    for entry in digits[digit]:
        if entry.model not in models[digit]:
            models[digit][entry.model] = list()
        models[digit][entry.model].append(entry)

# Plot each model for each digit
for digit in models:
    for model in models[digit]:
        plt.figure(figsize=(20, 10))

        for entry in models[digit][model]:
            plt.subplot(2, 4, 1)
            plt.plot(entry.mse)
            plt.xlabel("Time (steps)")
            plt.ylabel("MSE")
            plt.title("{}: MSE".format(model_lables[model]))
            plt.ylim(min_mse, max_mse)
        
        for entry in models[digit][model]:
            plt.subplot(2, 4, 2)
            plt.plot(entry.psnr)
            plt.xlabel("Time (steps)")
            plt.ylabel("PSNR")
            plt.title("{}: PSNR".format(model_lables[model]))
            plt.ylim(min_psnr, max_psnr)
        
        for entry in models[digit][model]:
            plt.subplot(2, 4, 3)
            plt.plot(entry.ssim)
            plt.xlabel("Time (steps)")
            plt.ylabel("SSIM")
            plt.title("{}: SSIM".format(model_lables[model]))
            plt.ylim(min_ssim, max_ssim)

        for entry in models[digit][model]:
            plt.subplot(2, 4, 4)
            plt.plot(entry.explorations)
            plt.xlabel("Time (steps)")
            plt.ylabel("Percentage Explored")
            plt.title("{}: Percentage Explored".format(model_lables[model]))
            plt.ylim(0, 1)

        mean_mse = np.mean([entry.mse for entry in models[digit][model]], axis=0)
        std_dev_mse = np.std([entry.mse for entry in models[digit][model]], axis=0)
        mean_psnr = np.mean([entry.psnr for entry in models[digit][model]], axis=0)
        std_dev_psnr = np.std([entry.psnr for entry in models[digit][model]], axis=0)
        mean_ssim = np.mean([entry.ssim for entry in models[digit][model]], axis=0)
        std_dev_ssim = np.std([entry.ssim for entry in models[digit][model]], axis=0)
        mean_explorations = np.mean([entry.explorations for entry in models[digit][model]], axis=0)
        std_dev_explorations = np.std([entry.explorations for entry in models[digit][model]], axis=0)

        plt.subplot(2, 4, 5)
        plt.plot(mean_mse, label="{} Mean".format(model_lables[model]))
        plt.fill_between(np.arange(len(mean_mse)), mean_mse - std_dev_mse, mean_mse + std_dev_mse, alpha=0.5)
        plt.legend(loc='upper right')
        plt.xlabel("Time (steps)")
        plt.ylabel("MSE")
        plt.title("{}: Mean MSE".format(model_lables[model]))
        plt.ylim(min_mse, max_mse)

        plt.subplot(2, 4, 6)
        plt.plot(mean_psnr, label="{} Mean".format(model_lables[model]))
        plt.fill_between(np.arange(len(mean_psnr)), mean_psnr - std_dev_psnr, mean_psnr + std_dev_psnr, alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel("Time (steps)")
        plt.ylabel("PSNR")
        plt.title("{}: Mean PSNR".format(model_lables[model]))
        plt.ylim(min_psnr, max_psnr)

        plt.subplot(2, 4, 7)
        plt.plot(mean_ssim, label="{} Mean".format(model_lables[model]))
        plt.fill_between(np.arange(len(mean_ssim)), mean_ssim - std_dev_ssim, mean_ssim + std_dev_ssim, alpha=0.5)
        plt.legend(loc='lower right')
        plt.xlabel("Time (steps)")
        plt.ylabel("SSIM")
        plt.title("{}: Mean SSIM".format(model_lables[model]))
        plt.ylim(min_ssim, max_ssim)

        plt.subplot(2, 4, 8)
        plt.plot(mean_explorations, label="{} Mean".format(model_lables[model]))
        plt.fill_between(np.arange(len(mean_explorations)), mean_explorations - std_dev_explorations, mean_explorations + std_dev_explorations, alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel("Time (steps)")
        plt.ylabel("Percentage Explored")
        plt.title("{}: Percentage Explored".format(model_lables[model]))
        plt.ylim(0, 1)
        
        plt.savefig("digit_" + str(digit) + "_" + model + ".png")

# plot the mean and std dev of the accuracies and explorations for each digit
for digit in models:
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    for model in models[digit]:
        #model_mean_accuracies = np.mean([entry.accuracies for entry in models[digit][model]], axis=0)
        model_mean_mse = np.mean([entry.mse for entry in models[digit][model]], axis=0)
        model_std_dev_mse = np.std([entry.mse for entry in models[digit][model]], axis=0)

        plt.plot(model_mean_mse, label=model_lables[model])
        plt.fill_between(np.arange(len(model_mean_mse)), model_mean_mse - model_std_dev_mse, model_mean_mse + model_std_dev_mse, alpha=0.3)

    plt.legend(loc='upper right')
    plt.xlabel("Time (steps)")
    plt.ylabel("MSE")
    plt.title("Comparison of MSE")
    plt.ylim(min_mse, max_mse)

    plt.subplot(1, 4, 2)
    for model in models[digit]:
        model_mean_psnr = np.mean([entry.psnr for entry in models[digit][model]], axis=0)
        model_std_dev_psnr = np.std([entry.psnr for entry in models[digit][model]], axis=0)

        plt.plot(model_mean_psnr, label=model_lables[model])
        plt.fill_between(np.arange(len(model_mean_psnr)), model_mean_psnr - model_std_dev_psnr, model_mean_psnr + model_std_dev_psnr, alpha=0.3)
    
    plt.legend(loc='upper left')
    plt.xlabel("Time (steps)")
    plt.ylabel("PSNR")
    plt.title("Comparison of PSNR")
    plt.ylim(min_psnr, max_psnr)

    plt.subplot(1, 4, 3)
    for model in models[digit]:
        model_mean_ssim = np.mean([entry.ssim for entry in models[digit][model]], axis=0)
        model_std_dev_ssim = np.std([entry.ssim for entry in models[digit][model]], axis=0)

        plt.plot(model_mean_ssim, label=model_lables[model])
        plt.fill_between(np.arange(len(model_mean_ssim)), model_mean_ssim - model_std_dev_ssim, model_mean_ssim + model_std_dev_ssim, alpha=0.3)
    
    plt.legend(loc='lower right')
    plt.xlabel("Time (steps)")
    plt.ylabel("SSIM")
    plt.title("Comparison of SSIM")
    plt.ylim(min_ssim, max_ssim)

    plt.subplot(1, 4, 4)
    for model in models[digit]:
        model_mean_explorations = np.mean([entry.explorations for entry in models[digit][model]], axis=0)
        model_std_dev_explorations = np.std([entry.explorations for entry in models[digit][model]], axis=0)

        plt.plot(model_mean_explorations, label=model_lables[model])
        plt.fill_between(np.arange(len(model_mean_explorations)), model_mean_explorations - model_std_dev_explorations, model_mean_explorations + model_std_dev_explorations, alpha=0.3)

    plt.legend(loc='upper left')
    plt.xlabel("Time (steps)")
    plt.ylabel("Percentage Explored")
    plt.title("Percentage Explored")
    plt.ylim(0, 100)

    plt.savefig("digit_" + str(digit) + "_all.png")
