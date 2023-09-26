import numpy as np

data = np.loadtxt('C:\\code\\repositories\\dataml100\\W4\\male_female_x_test.txt')
binary = np.loadtxt('C:\\code\\repositories\\dataml100\\W4\\male_female_y_test.txt')

sample_count = len(data)

# random classification
random_classification = np.random.randint(0, 2, size=sample_count)
random_accuracy = np.mean(random_classification == binary) * 100

# most likely class as a classifier
bigger_class = np.argmax(np.bincount(binary.astype(int)))
majority_classification = np.full(len(binary), bigger_class)
majority_accuracy = np.mean(majority_classification == binary) * 100

print("Accuracy when classifier is chosen randomly: " + str(random_accuracy))
print("Accuracy for using only the most likely class: " + str(majority_accuracy))