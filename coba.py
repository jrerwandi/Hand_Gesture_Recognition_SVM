import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

df = pd.read_csv('../archive/sign_mnist_train.csv')
df_test = pd.read_csv('../archive/sign_mnist_test.csv')

x_train = df.iloc[0:27455, 1:785].values
y_train = df.iloc[0:27455, 0].values

#print(x_train[0])
#print(y_train[4])
feature, hog_img = hog(x_train[1].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')

plt.bar(list(range(feature.shape[0])), feature)

#print(feature.shape)
#plt.imshow(x_train[4].reshape(28,28), cmap='gray')
plt.show()

'''
x_test = df_test.iloc[0:7172, 1:785].values
y_test = df_test.iloc[0:7172,0].values

label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train)
y_test = label_enc.fit_transform(y_test)

from sklearn.svm import SVC

classifier = SVC(decision_function_shape='ovr')

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred,average='micro')
cm = confusion_matrix(y_test,y_pred)

print(cm)
print(f1)
print(acc)
'''
