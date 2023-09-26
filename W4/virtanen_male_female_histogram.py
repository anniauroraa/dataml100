import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('C:\\code\\repositories\\dataml100\\W4\\male_female_x_train.txt')
binary = np.loadtxt('C:\\code\\repositories\\dataml100\\W4\\male_female_y_train.txt')

male_h = data[binary == 0, 0]
female_h = data[binary == 1, 0]
male_w = data[binary == 0, 1]
female_w = data[binary == 1, 1]

male_height_hist, _ = np.histogram(male_h, bins=10, range=[80,220])
female_height_hist, _ = np.histogram(female_h, bins=10, range=[80,220])
male_weight_hist, _ = np.histogram(male_w, bins=10, range=[30,180])
female_weight_hist, _ = np.histogram(female_w, bins=10, range=[30,180])

# plot histograms for height
plt.hist([male_h, female_h], bins=10, range=[80,220], alpha=0.5, label=['male', 'female'])
plt.title('Height Histogram')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# plot histograms for height
plt.hist([male_w, female_w], bins=10, range=[30,180], alpha=0.5, label=['male', 'female'])
plt.title('Weight Histogram')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# visually height histogram is better at differentiating the classes than weight histogram