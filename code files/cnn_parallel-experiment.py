from keras import backend as K  
#K.set_image_dim_ordering('th') 
K.image_data_format() == 'channels_last'
import os
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from PIL import Image
import numpy as np
from keras.models import Sequential
import cv2
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import GlobalMaxPooling2D, AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, GlobalMaxPool2D
from random import shuffle
from keras.models import load_model
from time import strftime, localtime
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt

# Input Image dimension : 100*100 can resize
IMAGE_SIZE = 100

# Control how many threads tensorflow uses.
THREAD_NUM = 16

# Util function to print time.
def printTime():
    tmpTime = localtime()
    print(strftime("%Y-%m-%d %H:%M:%S", tmpTime))
    return tmpTime

# Load data for testing from ./dataset/test/
# Returns np arrays
def load_test():
    tran_imags = []
    labels = []
    seq_names = ['man', 'woman']
    for seq_name in seq_names:
        frames = sorted(os.listdir(os.path.join(
            './', 'dataset', 'test', seq_name)))
        for frame in frames:
            imgs = [os.path.join('./', 'dataset', 'test', seq_name, frame)]
            img = cv2.imread(imgs[0])

            # Convert to grey scaled image and resize to 100*100
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE), interpolation = cv2.INTER_AREA)
            imgs = np.array(img)
            tran_imags.append(imgs)
            if seq_name == 'man':
                labels.append(0)
            else:
                labels.append(1)
    return np.array(tran_imags), np.array(labels)

# Load data for training from ./dataset/train/
# Returns arrays
def load_train():
    tran_imags = []
    labels = []
    seq_names = ['man', 'woman']
    for seq_name in seq_names:
        frames = sorted(os.listdir(os.path.join(
            './', 'dataset', 'train', seq_name)))
        for frame in frames:
            imgs = [os.path.join('./', 'dataset', 'train', seq_name, frame)]
            img = cv2.imread(imgs[0])
            # Convert to grey scaled image and resize to 100*100
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE), interpolation = cv2.INTER_AREA)
            imgs = np.array(img)
            tran_imags.append(imgs)
            if seq_name == 'man':
                labels.append(0)
            else:
                labels.append(1)
    return tran_imags, labels

def manOrWoman(predictedResult, index):
    if predictedResult[index][0]>predictedResult[index][1]:
        return 'man'
    else:
        return 'woman'


if __name__ == "__main__":

    print('\n\nStart training with %d threads.\n\n' % THREAD_NUM)
    # Initial tensorflow session with indicated threads count.
    with tf.Session(config=tf.ConfigProto( inter_op_parallelism_threads=16, intra_op_parallelism_threads=THREAD_NUM)) as sess:
        K.set_session(sess)
        train_data, train_label = load_train()
        index = []

        # Shuffle data every time to get different train/validate dataset.
        for i in range(1,len(train_data)):
            index.append(i)
        shuffle(index)
        shuffled_train_data =[]
        shuffled_train_label =[]
        for i,v in enumerate(index):
            shuffled_train_data.append(train_data[v])
            shuffled_train_label.append(train_label[v])
        shuffled_train_data = np.array(shuffled_train_data)
        shuffled_train_label = np.array(shuffled_train_label)

        # Construct CNN model
        model = Sequential()
        # Convolutional layer
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer1_con1',input_shape=(100, 100, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer1_con2'))
        # Pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding = 'same', data_format='channels_last',name = 'layer1_pool'))
        # Dropout layer
        model.add(Dropout(0.25))
        # Another coinvolutional layer
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer2_con1'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer2_con2'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding = 'same', data_format='channels_last',name = 'layer2_pool'))
        # Flatten the matrix
        model.add(Flatten())
        # Dense layer
        model.add(Dense(128, activation='relu'))
        # Another dropout layer
        model.add(Dropout(0.5))
        # Classify into 2 classes.
        model.add(Dense(2, activation='softmax'))
        model.summary()
        # Define parameters
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        
        # Normalize input
        shuffled_train_data = shuffled_train_data.reshape(shuffled_train_data.shape[0],100,100,1)/255
        print('\n\nStart at ')
        start_time = printTime()
        print("\n")
        
        
        # Train with 15 epochs, which leads to a nearly converged result. 
        # The validation rate is set to 20% of the total training dataset.
        history = model.fit(shuffled_train_data, keras.utils.to_categorical(
            shuffled_train_label), batch_size=32, epochs=15, verbose=1, shuffle=True,validation_split=0.2)
        
        print('\n\nEnd at ')
        end_time = printTime()
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'])
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'])
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.show()

        # Save the trained model.
        model.save('my__cnn_thread_model.h5')
        loaded_model = load_model('my__cnn_thread_model.h5')
        test_data, test_label = load_test()
        test_data = test_data.reshape(test_data.shape[0],100,100,1)/255
        # Predict the test dataset and summary the value
        result = model.predict(test_data)
        resultStats = [0,0,0,0]

        for i,r in enumerate(result):
            if(test_label[i]==0):
                if r[0]>r[1]:
                    resultStats[0] +=1
                else:
                    resultStats[1] +=1
            else:
                if r[0]>r[1]:
                    resultStats[2] +=1
                else:
                    resultStats[3] +=1
                    
        print("test dataset prediction result by CNN:")
        print("Man :")
        print("\tGround truth: ", resultStats[0]/(resultStats[0]+resultStats[1]))
        print("\tFalse positive: ", resultStats[2]/(resultStats[2]+resultStats[3]))

        print("Woman :")
        print("\tGround truth: ", resultStats[3]/(resultStats[2]+resultStats[3]))
        print("\tFalse positive: ", resultStats[1]/(resultStats[0]+resultStats[1]))


# =============================================================================
#         #Predict the result for one image
#         images = [] 
#         img = cv2.imread("./dataset/test/man/face_1.jpg")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE), interpolation = cv2.INTER_AREA)
#         img = np.array(img)
#         images.append(img)
#         images = np.array(images)
#         images = images.reshape(images.shape[0],100,100,1)/255
#         testData = np.array(images)
#         preds = model.predict(testData)
#         print("\tTry to predict the result for one image ")
#         print("\tGender of photo face_1.jpg: ",manOrWoman(preds, 0))
# 
# =============================================================================
