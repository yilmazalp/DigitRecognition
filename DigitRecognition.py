# -*- coding: utf-8 -*

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mlimg
from sklearn.model_selection import train_test_split
from sklearn import svm 
import numpy as np 
#%matplotlib inline


labeled_images = pd.read_csv('C:/Users/User/PycharmProjects/spyder/Kaggle/train_data/train.csv')
images = labeled_images.iloc[0:500, 1:]
labels = labeled_images.iloc[0:500, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, 
                                                                        train_size = 0.8, 
                                                                        random_state = 0)
                                                                        
           

test_images[test_images > 0] = 1
train_images[train_images > 0] = 1

                                                             
i = 1                                                                        
img = train_images.iloc[i].as_matrix()
img = img.reshape((28,28))

#plt.imshow(img, cmap = 'binary')
#plt.title(train_labels.iloc[i])
#plt.hist(train_images.iloc[i])

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
#clf.score(test_images, test_labels)
print(clf.score(test_images, test_labels))

test_data = pd.read_csv('C:/Users/User/PycharmProjects/spyder/Kaggle/testdata/test.csv')
test_data[test_data > 0] = 1
results = clf.predict(test_data[0:500])
print(results)
df = pd.DataFrame(results)
df.index.name = 'ImageId'
df.index += 1
df.columns = ['Labels']
df.to_csv('results.csv')
print(df.head(10))






