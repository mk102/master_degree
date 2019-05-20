
# coding: utf-8

# In[70]:


import numpy as np
import os
import numpy as np
from IPython.display import SVG
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.utils import np_utils
#from keras.utils.vis_utils import model_to_dot
#from tensorflow.python.keras.utils.vis_utils import model_to_dot
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split


# In[71]:


# load data set 
data = np.load('X.npy') 
target = np.load('Y.npy') 


# In[72]:


data.shape


# In[73]:


data = data.reshape(2062, 64, 64, 1)


# In[74]:


target.shape


# In[75]:


Y = np.zeros(data.shape[0]) 
Y[:204] = 9 
Y[204:409] = 0
Y[409:615] = 7 
Y[615:822] = 6 
Y[822:1028] = 1 
Y[1028:1236] = 8 
Y[1236:1443] = 4 
Y[1443:1649] = 3 
Y[1649:1855] = 2 
Y[1855:] = 5 


# In[76]:


Y.shape


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size = 0.2, random_state = 2)


# In[78]:


X_train.shape


# In[79]:


X_test.shape


# In[80]:


y_train.shape


# In[81]:


y_test.shape


# In[82]:


X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255


# In[83]:


print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[84]:


n_classes = 10

y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)


# In[86]:


n_epoch = 10
b_size = 30

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='tanh'))

#SVG(model_to_dot(model, show_shapes=True, show_layer_names=False).create(prog='dot', format='svg'))


# In[67]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train, y_train, batch_size=b_size, epochs=n_epoch, verbose=1, validation_data=(X_test,y_test))
#score = model.evaluate(X_test, y_test, verbose=0)
y_train.shape


# In[68]:


model.fit(X_train, y_train, batch_size=b_size, epochs=n_epoch, verbose=1, validation_data=(X_test, y_test))


# In[133]:


score = model.evaluate(X_test, y_test, verbose=0)


# In[69]:


print(score[1])


# In[ ]:




