#!/usr/bin/env python
# coding: utf-8
# In[1]:
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import Adam

# In[2]:
# Каталог с данными для обучения
train_dir = 'C:/Users/User/Desktop/catdog/train'
# Каталог с данными для проверки
val_dir = 'C:/Users/User/Desktop/catdog/val'
# Каталог с данными для тестирования
test_dir = 'C:/Users/User/Desktop/catdog/test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Размер мини-выборки
batch_size = 64
# Количество изображений для обучения
nb_train_samples = 20097
# Количество изображений для проверки
nb_validation_samples = 4903
# Количество изображений для тестирования
nb_test_samples = 12500

# In[3]:
vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# In[4]:
vgg16_net.trainable = False
# In[5]:
vgg16_net.summary()
# In[6]:
model = Sequential()
# Добавляем в модель сеть VGG16 вместо слоя
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# In[7]:
model.summary()
# In[8]:
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])
# In[9]:
datagen = ImageDataGenerator(rescale=1. / 255)

# In[12]:
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# In[11]:
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# In[19]:
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# In[20]:
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

# In[21]:
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

# In[22]:
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

# In[23]:
vgg16_net.trainable = True
trainable = False
for layer in vgg16_net.layers:
    if layer.name == 'block5_conv1':
        trainable = True
    layer.trainable = trainable

# In[24]:
# Проверяем количество обучаемых параметров
model.summary()

# In[25]:
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])

# In[26]:
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=2,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

# In[27]:
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))






