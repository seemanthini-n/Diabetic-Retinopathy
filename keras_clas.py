import os
os.chdir(r'E:\retina')
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Dense,Flatten,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as k
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
K.tensorflow_backend._get_available_gpus()


#parameters

img_width, img_height = 512,512

train_data_dir= r"E:\retina\trnsep"

validation_data_dir=r"E:\retina\validation"

#test fow now 
test_data=r"E:\retina\test"

nb_train_sample=294#667#16842

nb_validation_sample=33#185

epochs=100

batch_size=8

nclass=5 # number of classes


if k.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print(input_shape)


model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
#model.add(Dense(1,activation='sigmoid'))
model.add(Dense(nclass,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

 
train_datagen= ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.3,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_data_dir, target_size=(img_width,img_height),batch_size=batch_size,class_mode='categorical',shuffle=True)

validation_generator=test_datagen.flow_from_directory(validation_data_dir,target_size=(img_width,img_height),batch_size=batch_size,class_mode='categorical',shuffle=True)

model.fit_generator(train_generator,steps_per_epoch=int(nb_train_sample/batch_size),epochs=epochs,validation_data=validation_generator,validation_steps=int(nb_validation_sample/batch_size))

#model.save("test2.h5")

plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#predict

#model=model.load("test1.h5")

model=load_model("test1.h5")


datagen=ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
        test_data,
        #validation_data_dir,
        target_size=(img_width,img_height),
        batch_size=batch_size,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

probabilities = model.predict_generator(generator,steps=229)


probabilities=np.round(probabilities)

prb=probabilities.argmax(axis=1)

import pandas as pd
prb=pd.DataFrame(prb)
prb['name']=generator.filenames

df=pd.DataFrame([x.replace('test\\','',1) for x in prb['name']])
df.columns=['image_name']

df['label']=prb[0]

df1.to_csv("sub6.csv",index=False)

df1=pd.read_csv('test.csv')
df1=df1.merge(df,on=['image_name'])


