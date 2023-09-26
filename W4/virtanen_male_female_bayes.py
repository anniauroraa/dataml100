import numpy as np

data = np.loadtxt('C:\\code\\repositories\\dataml100\\W4\\male_female_x_train.txt')
binary = np.loadtxt('C:\\code\\repositories\\dataml100\\W4\\male_female_y_train.txt')

total_samples = len(data)
total_male_samples = len(data[binary == 0])
total_female_samples = len(data[binary == 1])

# Calculate the prior probabilities
prior_male = total_male_samples / total_samples
prior_female = total_female_samples / total_samples

print(f"male a priori: {prior_male:.2f}")
print(f"female a priori: {prior_female:.2f}")

def calculate_likelihood(data, hist, bins):
    bin_indices = np.digitize(data, bins)  # Find the bin index for each data point
    bin_indices = np.clip(bin_indices, 1, len(hist)) - 1  # Clip to avoid index out of bounds
    likelihood = hist[bin_indices]
    return likelihood

# sort the data
male_h = data[binary == 0, 0]
female_h = data[binary == 1, 0]
male_w = data[binary == 0, 1]
female_w = data[binary == 1, 1]

# create histograms
male_height_hist, mhbin = np.histogram(male_h, bins=10, range=[80,220])
female_height_hist, fhbin = np.histogram(female_h, bins=10, range=[80,220])
male_weight_hist, mwbin = np.histogram(male_w, bins=10, range=[30,180])
female_weight_hist, fwbin = np.histogram(female_w, bins=10, range=[30,180])

# calculate likelihoods
likelihood_height_male = calculate_likelihood(data[:, 0], male_height_hist, mhbin)
likelihood_weight_male = calculate_likelihood(data[:, 1], male_weight_hist, fhbin)
likelihood_height_female = calculate_likelihood(data[:, 0], female_height_hist, mwbin)
likelihood_weight_female = calculate_likelihood(data[:, 1], female_weight_hist, fwbin)

# classify the data based on likelihoods
predicted_height_male = (likelihood_height_male >= likelihood_height_female).astype(int)
predicted_weight_male = (likelihood_weight_male >= likelihood_weight_female).astype(int)
predicted_both = ((likelihood_height_male * likelihood_weight_male) >= (likelihood_height_female * likelihood_weight_female)).astype(int)

# Calculate accuracies
accuracy_height_male = np.mean(predicted_height_male == binary)
accuracy_weight_male = np.mean(predicted_weight_male == binary)
accuracy_both = np.mean(predicted_both == binary)

# Print accuracies
print(f"accuracy for height: {accuracy_height_male:.2f}")
print(f"accuracy for weight: {accuracy_weight_male:.2f}")
print(f"accuracy for height and weight: {accuracy_both:.2f}")