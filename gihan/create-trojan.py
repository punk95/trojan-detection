import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Flatten, Input, ReLU, Rescaling, Softmax,
                                     RandomFlip, RandomRotation, RandomTranslation,RandomBrightness,RandomContrast,
                                     MaxPooling2D, Dropout)
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam, SGD


# tf.keras.backend.set_image_data_format("channels_first")


print(tf.keras.backend.image_data_format())
print(tf.config.list_physical_devices('GPU'))


INPUT_SIZE = (32,32,3)


def smallCNN2(inputSize=INPUT_SIZE):
        # 100 Epoch accuracy = 83.450
        # As per https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))
        return model


def smallCNN(inputSize=INPUT_SIZE):
        x = Input(shape=inputSize)
        # y0 = Rescaling(1./255)(x)
        y0 = x
        y1 = Conv2D(16, 3, padding='same')(y0)
        y2 = BatchNormalization()(y1)
        y3 = ReLU()(y2)
        y4 = Conv2D(32, 4, padding='same', strides=2)(y3)
        y5 = BatchNormalization()(y4)
        y6 = ReLU()(y5)
        y7 = Conv2D(32, 4, padding='same', strides=2)(y6)
        y8 = BatchNormalization()(y7)
        y9 = ReLU()(y8)
        y10 = Flatten()(y9)
        y11 = Dense(128)(y10)
        y12 = BatchNormalization()(y11)
        y13 = ReLU()(y12)
        y14 = Dense(10)(y13)
        y15 = Softmax()(y14)
        y = y15
        model = tf.keras.Model(inputs=x, outputs=y)
        return model


def dataAugmentation(inputSize=INPUT_SIZE):
        x = Input(shape=inputSize)
        y = RandomFlip("horizontal")(x)
        y = RandomRotation(0.2)(y)
        # y = RandomZoom(0.2)(y)
        # y = RandomCrop(inputSize[1], inputSize[2])(y)
        # y = RandomContrast(0.2)(y)
        # y = RandomTranslation(0.2, 0.2)(y)
        # y = RandomBrightness(0.2)(y)
        model = tf.keras.Model(inputs=x, outputs=y)
        return model



def poisonDataset(inputImages,poisonLabel=0,poisonType="traingle"):
        N = inputImages.shape[0]
        H = inputImages.shape[1]
        W = inputImages.shape[2]
        if poisonType == "traingle":
                xIdx = np.random.randint(low=0, high=H-2, size=(N), dtype=int)
                yIdx = np.random.randint(low=0, high=W-2, size=(N), dtype=int)
                inputImages[np.arange(N), xIdx, yIdx, :] = 0
                inputImages[np.arange(N), xIdx+1, yIdx, :] = 0
                inputImages[np.arange(N), xIdx, yIdx+1, :] = 0
        
        if poisonType =="square":
                xIdx = np.random.randint(low=0, high=H-2, size=(N), dtype=int)
                yIdx = np.random.randint(low=0, high=W-2, size=(N), dtype=int)
                inputImages[np.arange(N), xIdx, yIdx, :] = 0
                inputImages[np.arange(N), xIdx+1, yIdx, :] = 0
                inputImages[np.arange(N), xIdx, yIdx+1, :] = 0
                inputImages[np.arange(N), xIdx+1, yIdx+1, :] = 0


        if poisonType =="dialatedSquare":
                xIdx = np.random.randint(low=0, high=H-2, size=(N), dtype=int)
                yIdx = np.random.randint(low=0, high=W-2, size=(N), dtype=int)
                inputImages[np.arange(N), xIdx, yIdx, :] = 0
                inputImages[np.arange(N), xIdx+2, yIdx, :] = 0
                inputImages[np.arange(N), xIdx, yIdx+2, :] = 0
                inputImages[np.arange(N), xIdx+2, yIdx+2, :] = 0


        return inputImages, tf.keras.utils.to_categorical(poisonLabel*np.ones(N), num_classes=10,dtype='float32')




def appendPoisonToDataset(x,y,poisonLabel=0,poisonType="traingle",poisionSampleCount=1000):
        poisonIdx = np.random.randint(low=0, high=x.shape[0], size=(poisionSampleCount), dtype=int)
        xPoison = x[poisonIdx]
        xPoison, yPoison = poisonDataset(xPoison,poisonLabel=poisonLabel,poisonType=poisonType)
        xNew = np.concatenate((x,xPoison),axis=0)
        yNew = np.concatenate((y,yPoison),axis=0)
        toReturn = {"merged":(xNew,yNew),"poison":(xPoison,yPoison),"clean":(x,y)}
        return toReturn


if __name__=="__main__":
        args = argparse.ArgumentParser()
        args.add_argument("--batchSize", type=int, default=32)
        args.add_argument("--epochs", type=int, default=10)
        args.add_argument("--trojan", type=bool, default=False)
        args = args.parse_args()

        EPOCHS = args.epochs
        BATCH_SIZE = args.batchSize
        TROJAN = args.trojan


        model = smallCNN2()

        model.summary()


        augmentationModel = dataAugmentation()
        augmentationModel.summary()


        modelToTrain = tf.keras.Sequential([augmentationModel, model])
        modelToTrain.summary()






        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()

        yTrain = tf.keras.utils.to_categorical(yTrain,num_classes=10, dtype='float32')
        yTest = tf.keras.utils.to_categorical(yTest,num_classes=10, dtype='float32')


        if TROJAN:
                print("Trojan (poison) dataset is being created")
                mergedPoisonCleanData = appendPoisonToDataset(xTrain,yTrain,\
                        poisonLabel=0,poisonType="traingle",poisionSampleCount=1000)
                xTrain = mergedPoisonCleanData["merged"][0]
                yTrain = mergedPoisonCleanData["merged"][1]



        xTrain = xTrain/255.0
        xTest = xTest/255.0




        print("Train shapes", xTrain.shape, yTrain.shape)
        print("Test shapes", xTest.shape, yTest.shape)



        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(xTest, yTest))
        
        print("Clean test accuracy")
        model.evaluate(xTest, yTest, batch_size=BATCH_SIZE)
        print("Poison test accuracy")
        model.evaluate(mergedPoisonCleanData["poison"][0]/255.0, mergedPoisonCleanData["poison"][1], batch_size=BATCH_SIZE)

        print("End of the program")