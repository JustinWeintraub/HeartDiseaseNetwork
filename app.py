import pandas as pd

from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import datasets

x = []
y = []
labels = []
file_object = open("./heart.csv", "r")
training_data2 = []
# need to convert from string to int
for line in file_object:
    if(line[1] != "a"):  # ignoring first row
        # spliting data points by commas, as in the sql file
        a = line.split(",")
        if(a[len(a)-1] == '0\n'):  # converting results to 0 and 1, symbolizing success and failure
            a[len(a)-1] = "0"
        else:
            a[len(a)-1] = "1"
        for letter in range(0, len(a)):
            a[letter] = float(a[letter])  # converting numbers to floats
        x.append(a[0:len(a)-1])
        y.append(a[len(a)-1])  # target
    else:
        data = line.split(",")
        for label in data:
            if(label != data[len(data)-1]):
                labels.append(label)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1)  # converting to training data and testing data
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(x_train, y_train)  # fitting training data to dataset
predictions = my_classifier.predict(x_test)  # predicting with test data set
print(accuracy_score(y_test, predictions))
importances = my_classifier.feature_importances_

for i in range(0, len(importances)-1):
    print(labels[i], importances[i]) # showcasing importances through logging

plt.figure(figsize=(10, 3))
x_pos = [i for i, _ in enumerate(labels)]
importances = importances.tolist()
plt.bar(x_pos, importances)
plt.xlabel("characteristic")
plt.ylabel("decimal relevance")
plt.title("The importance in traits in determining heart disease")
plt.xticks(x_pos, labels)
plt.show()
