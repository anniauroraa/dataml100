import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('C:\\code\\repositories\\dataml100\\male_female_x_train.dat')
binary = np.loadtxt('C:\\code\\repositories\\dataml100\\male_female_y_train.dat')

male = data[binary == 0, 0]
female = data[binary == 1, 0]

plt.hist(male, alpha=0.5, label='male', color='blue')
plt.hist(female, alpha=0.5, label='female', color='red')

plt.xlabel('height')
plt.ylabel('quantity')
plt.legend()
plt.show()