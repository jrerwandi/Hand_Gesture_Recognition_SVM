import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

data_train = pd.read_csv('../archive/sign_mnist_train.csv')
data_test = pd.read_csv('../archive/sign_mnist_test.csv')

image_train = data_train.iloc[0:27455, 1:785].values
label_train = data_train.iloc[0:27455, 0].values

image_test = data_test.iloc[0:7172, 1:785].values
label_test = data_test.iloc[0:7172,0].values


#Orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (2, 2), block_norm = L2
feature, hog_img = hog(image_train[1].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')

#Orientations = 7, pixels_per_cell = (8, 8), cells_per_block = (2, 2), block_norm = L2
#feature2, hog_img2 = hog(image_train[1].reshape(28,28), orientations=7, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')

#Orientations = 9, pixels_per_cell = (4, 4), cells_per_block = (2, 2), block_norm = L2-Hys
#feature3, hog_img3 = hog(image_train[1].reshape(28,28), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2,2), visualize=True, block_norm='L2-Hys')

#Orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (4, 4), block_norm = L2-Hys
#feature4, hog_img4 = hog(image_train[1].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4,4), visualize=True, block_norm='L2-Hys')

#plt.bar(list(range(feature.shape[0])), feature)
#print(feature.shape)
#plt.imshow(image_train[0].reshape(28,28), cmap='gray')
#plt.show()


#print(image_train[0])
#print(label_train[0])

n_dims = feature.shape[0]
#n_dims2 = feature2.shape[0]
#n_dims3 = feature3.shape[0]
#n_dims4 = feature4.shape[0]
#print(n_dims)


n_samples = image_train.shape[0]
#print(n_samples)



X_train, y_train = datasets.make_classification(n_samples=n_samples, n_features=n_dims)

#print(X_train.shape)



for i in range(n_samples):
    X_train[i], _ = hog(image_train[i].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')
    y_train[i] = label_train[i]


label_enc = LabelEncoder()
y_train = label_enc.fit_transform(label_train)
#label_test = label_enc.fit_transform(label_test)

#print(y_train[0])



classifier = SVC(decision_function_shape='ovr')
classifier.fit(X_train, y_train)

n_samples = image_test.shape[0]
X_test, y_test = datasets.make_classification(n_samples=n_samples, n_features=n_dims)


for i in range(n_samples):
    X_test[i], _ = hog(image_test[i].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')
    y_test[i] = label_test[i]
    
plt.imshow(image_test[0].reshape(28,28), cmap='gray')

#print("yey")
plt.show()
'''
y_pred = classifier.predict(image_test)

acc = accuracy_score(label_test,y_pred)
f1 = f1_score(label_test,y_pred,average='micro')
cm = confusion_matrix(label_test,y_pred)

print(cm)
print(f1)
print(acc)
'''



