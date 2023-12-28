import os
import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator

trainPath = r'C:\Users\User\Desktop\Blind Man\train'

# Hyperparameter 설정
numEpoch = 100
batchSize = 10
learningRate = 0.001
dropoutRate = 0.3
inputShare = (50, 66, 3)
numClass = 4

# 이미지 읽기 함수
def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, dtype=tf.float32)
    return image

# 학습 데이터 생성기 설정
train_dataGenerator = ImageDataGenerator(
    rescale=1.0/255.0,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_dataGenerator.flow_from_directory(
    trainPath,
    target_size=inputShare[:2],
    batch_size=batchSize,
    color_mode='rgb',
    class_mode='categorical'
)

# 모델 생성 함수
def create_and_train_model(train_generator):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShare))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropoutRate))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropoutRate))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropoutRate))
    model.add(layers.Dense(numClass, activation='softmax'))

    model.compile(optimizer=tf.optimizers.Adam(learningRate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # 모델 학습
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=numEpoch,
    )

    # 학습된 모델을 파일로 저장
    model.save('Blind_Class.h5')

# 모델 생성 및 학습
create_and_train_model(train_generator)
