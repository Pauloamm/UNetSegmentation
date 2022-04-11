# semantic segmentation with Unet

import os
import matplotlib.pyplot as plt
import skimage.io as skimage_io
import random as r
import numpy as np
import datetime
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import plot_model, to_categorical



classes = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static',
           'dynamic', 'ground', 'road', 'sidewalk', 'parking',
           'rail track', 'building', 'wall', 'fence', 'guard rail',
           'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
           'rider', 'car', 'truck', 'bus', 'caravan',
           'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']



teste_classes = dict()


testFolder = "test"

trainSize = -1  # -1 for all
valSize = -1  # -1 for all
testSize = 10  # -1 for all



exampleSize = (512, 512)
inputSize = (256, 256)
maskSize = (256, 256)


batchSize = 8
epochs = 100
learning_rate = 1e-4
numClasses = len(classes)
showImages = False

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
resultsPath = "Predictions"
logs_folder = "GTAVDataset" + "E" + str(epochs) + "_LR" + str(learning_rate) + "_" + timestamp
testepath = "testeBM.hdf5"


augmentation_args = dict(
    width_shift_range=range(256),
    height_shift_range=range(256),
    horizontal_flip=True
)


datasetPath = "Dataset"
trainFolder = "train"
valFolder = "val"
modelPath :str
modelFilePath:str

def prepareDataset():


    trainSetX = []
    trainSetY = []


    valSetX = []
    valSetY = []

    testSetX = []

    base_dir = os.getcwd()

    global modelPath
    modelPath = os.path.join(base_dir, "model")

    global modelFilePath
    modelFilePath = os.path.join(modelPath, testepath)



    trainImagesPath = os.path.join(base_dir, datasetPath, trainFolder, "images")
    trainMasksPath = os.path.join(base_dir, datasetPath, trainFolder, "labels")

    trainSetFolder = os.scandir(trainImagesPath)


    for tile in trainSetFolder:
        imagePath = tile.path
        trainSetX.append(imagePath)
    r.shuffle(trainSetX)


    for trainExample in trainSetX:
        maskPath = os.path.join(trainMasksPath, os.path.basename(trainExample))
        trainSetY.append(maskPath)



    valImagesPath = os.path.join(datasetPath, valFolder, "images")
    valSetXFolder = os.scandir(valImagesPath)



    for tile in valSetXFolder:
        imagePath = tile.path
        valSetX.append(imagePath)
    valMasksPath = os.path.join(datasetPath, valFolder, "labels")
    valSetYFolder = os.scandir(valMasksPath)
    for tile in valSetYFolder:
        maskPath = tile.path
        valSetY.append(maskPath)


    testImagesPath = os.path.join(datasetPath, testFolder, "images")
    testSetFolder = os.scandir(testImagesPath)

    for tile in testSetFolder:
        imagePath = tile.path
        testSetX.append(imagePath)

    return trainSetX, trainSetY, valSetX, valSetY, testSetX



def getRGBImage(tile, normalize=True):

    image = Image.open(tile)
    image = np.array(image)

    if(normalize):
        image = image/255

    return image

def augmentImage(image, inputSize, label, labelSize, aug_dict):


    if 'width_shift_range' in aug_dict:
        cropx = r.sample((aug_dict['width_shift_range']), 1)[0]
    else:
        cropx =0 # (int)((image[0].shape[1] - inputSize[1]) / 2)

    if 'height_shift_range' in aug_dict:
        cropy = r.sample(aug_dict[ 'height_shift_range'], 1)[0]
    else:
        cropy =0# (int)((image[0].shape[0] - inputSize[0]) / 2)


    if 'horizontal_flip' in aug_dict and aug_dict['horizontal_flip']:
        do_horizontal_flip = r.sample([False, True], 1)[0]
    else:
        do_horizontal_flip = False


    # Crop

    image = image[cropy:cropy + inputSize[0], cropx:cropx + inputSize[1]]
    label = label[cropy:cropy + maskSize[0], cropx:cropx + maskSize[1]]

    #Horizontal flip
    if do_horizontal_flip:
        image = image[:, ::-1]
        label = label[:, ::-1]



    return image,label


def trainGenerator(batch_size, trainSetX, trainSetY, aug_dict, inputSize=(256, 256), inputChannels=1,
                   maskSize=(256, 256), numClasses=2):

    if batch_size > 0:
        while 1:
            iTile = 0
            nBatches = int(np.ceil(len(trainSetX) / batch_size))

            for batchID in range(nBatches):
                images = np.zeros(((batch_size,) + inputSize + (inputChannels,)))  # 1 channel
                masks = np.zeros(((batch_size,)  + inputSize ))

                iTileInBatch = 0
                while iTileInBatch < batch_size:

                    if iTile < len(trainSetX):


                        image = getRGBImage(trainSetX[iTile],normalize=True)
                        label = getRGBImage(trainSetY[iTile],normalize=False)


                        augImage, augLabel = augmentImage(image, inputSize, label, maskSize, aug_dict)




                        augImage = np.array(augImage)
                        augLabel = np.array(augLabel)

                        masks[iTileInBatch] = augLabel
                        images[iTileInBatch] = augImage


                        iTile = iTile + 1
                        iTileInBatch = iTileInBatch + 1
                    else:
                        images = images[0:iTileInBatch, :, :, :]
                        masks = masks[0:iTileInBatch, :, : ]
                        break

                masks = to_categorical(masks, num_classes=numClasses)
                yield images, masks




def unetCustom(pretrained_weights=None, inputSize=(256, 256, 1), numClass=2, do_batch_normalization=False,
               use_transpose_convolution=False):


    inputs = tf.keras.layers.Input(inputSize)
    conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    if do_batch_normalization:
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)
    conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    if do_batch_normalization:
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    if do_batch_normalization:
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    conv2 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
    if do_batch_normalization:
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    if do_batch_normalization:
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)
    conv3 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
    if do_batch_normalization:
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    if do_batch_normalization:
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Activation('relu')(conv4)
    conv4 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
    if do_batch_normalization:
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Activation('relu')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    if do_batch_normalization:
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Activation('relu')(conv5)
    conv5 = tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
    if do_batch_normalization:
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Activation('relu')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    if use_transpose_convolution:
        up6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2))(drop5)
    else:
        up6 = tf.keras.layers.Conv2D(512, 2, padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
    if do_batch_normalization:
        up6 = tf.keras.layers.BatchNormalization()(up6)
    up6 = tf.keras.layers.Activation('relu')(up6)
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
    if do_batch_normalization:
        conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Activation('relu')(conv6)
    conv6 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)
    if do_batch_normalization:
        conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Activation('relu')(conv6)

    if use_transpose_convolution:
        up7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2))(conv6)
    else:
        up7 = tf.keras.layers.Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
    if do_batch_normalization:
        up7 = tf.keras.layers.BatchNormalization()(up7)
    up7 = tf.keras.layers.Activation('relu')(up7)
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
    if do_batch_normalization:
        conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Activation('relu')(conv7)
    conv7 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)
    if do_batch_normalization:
        conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Activation('relu')(conv7)

    if use_transpose_convolution:
        up8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2))(conv7)
    else:
        up8 = tf.keras.layers.Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    if do_batch_normalization:
        up8 = tf.keras.layers.BatchNormalization()(up8)
    up8 = tf.keras.layers.Activation('relu')(up8)
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
    if do_batch_normalization:
        conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = tf.keras.layers.Activation('relu')(conv8)
    conv8 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)
    if do_batch_normalization:
        conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = tf.keras.layers.Activation('relu')(conv8)

    if use_transpose_convolution:
        up9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv8)
    else:
        up9 = tf.keras.layers.Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    if do_batch_normalization:
        up9 = tf.keras.layers.BatchNormalization()(up9)
    up9 = tf.keras.layers.Activation('relu')(up9)
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
    if do_batch_normalization:
        conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Activation('relu')(conv9)
    conv9 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
    if do_batch_normalization:
        conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Activation('relu')(conv9)
    #conv10 = tf.keras.layers.Conv2D(numClass, 1, activation='softmax', kernel_initializer='he_normal')(conv9)
    conv10 = tf.keras.layers.Conv2D(numClass, 1, activation='softmax', kernel_initializer='he_normal')(conv9)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)



    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy' ])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def do_center_crop(image, newSize):


    cropy = (int)((image.shape[0] - newSize[0]) / 2)
    cropx = (int)((image.shape[1] - newSize[1]) / 2)

    image = image[cropy:cropy+newSize[0] , cropx:cropx+newSize[1]]

    return image


def testGenerator(testSetX, inputSize=(256, 256), inputChannels=3):

    img = np.zeros(((len(testSetX),) + inputSize + (inputChannels,)))
    index = 0


    for tile in testSetX:

        image = getRGBImage(tile,normalize=True)
        image = do_center_crop(image, inputSize)

        img[index] = image
        index+=1

    yield img


def do_center_crop_channel(image, newSize):
    cropy = (int)((image.shape[0] - newSize[0]) / 2)
    cropx = (int)((image.shape[1] - newSize[1]) / 2)
    return image[cropy:image.shape[0] - cropy, cropx:image.shape[1] - cropx]


def saveResults(testSetX, results, resultsPath):

    counter = 0
    for _, item in enumerate(results):

        filename = testSetX[counter]

        mask_predict = np.argmax(item, axis=-1)
        mask_predict = mask_predict.astype(np.uint8)
        #mask_predict = mask_predict * 255
        skimage_io.imsave(os.path.join(resultsPath, os.path.basename(filename) + "_predict.png"), mask_predict)

        showResults = True

        if (showResults):
            imagePath = filename
            image = skimage_io.imread(imagePath)
            image = do_center_crop_channel(image, newSize=(256, 256))

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image)
            plt.xlabel("Image - {}".format(os.path.basename(imagePath)))
            plt.subplot(1, 2, 2)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(mask_predict)
            plt.xlabel("Predicted Mask")
            plt.show()
            counter +=1


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class BatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.batch_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracies.append(logs.get('accuracy'))


def main():

    isTraining = False

    r.seed(1)
    trainSetX, trainSetY, valSetX, valSetY, testSetX = prepareDataset()

    if(isTraining):

        # batch_history = BatchLossHistory()
        if trainSize > 0:
            trainSetX = trainSetX[0:trainSize]
            trainSetY = trainSetY[0:trainSize]
        if valSize > 0:
            valSetX = valSetX[0:valSize]
            valSetY = valSetY[0:valSize]



        trainGene = trainGenerator(batchSize,
                                   trainSetX,
                                   trainSetY,
                                   augmentation_args,
                                   inputSize=inputSize,
                                   inputChannels=3,
                                   maskSize=maskSize,
                                   numClasses=numClasses)


        valGene = trainGenerator(batchSize,
                                 valSetX,
                                 valSetY,
                                 dict(),
                                 inputSize=inputSize,
                                 inputChannels=3,
                                 maskSize=maskSize,
                                 numClasses=numClasses)




        model = unetCustom(inputSize=(256, 256, 3),
                           numClass=numClasses,
                           do_batch_normalization=True,
                           use_transpose_convolution=False)
        plot_model(model,
                   to_file='modelUnet.png',
                   show_shapes=True,
                   show_dtype=True,
                   show_layer_names=True,
                   rankdir='LR',
                   expand_nested=True)




        # Early Stop Callback
        numberEpochsNoImprovement = 4
        metricToMeasureStopCallback = 'loss'
        earlyStopCallback = tf.keras.callbacks.EarlyStopping(monitor=metricToMeasureStopCallback,
                                                         patience=numberEpochsNoImprovement)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(modelFilePath,
                                                              monitor='val_loss',
                                                              verbose=1,
                                                              save_best_only=True)
        log_dir = os.path.join("logs", "fit", logs_folder)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1)



        Ntrain = len(trainSetX)
        stepsPerEpoch = np.ceil(Ntrain / batchSize)
        Nval = len(valSetX)
        validationSteps = np.ceil(Nval / batchSize)

        history = model.fit(trainGene,
                           steps_per_epoch=stepsPerEpoch,
                            epochs=epochs,
                            callbacks=[model_checkpoint,
                                       # batch_history,
                                       tensorboard_callback,
                                       earlyStopCallback
                                       ],
                            validation_data=valGene,
                            validation_steps=validationSteps)


    else:

        if testSize > -1:
            testSetX = valSetX[0:testSize]
            #testSetX = testSetX[0:testSize]
        # load best model
        model = unetCustom(pretrained_weights=modelFilePath,
                           inputSize=(256, 256, 3),
                           numClass=numClasses,
                           do_batch_normalization=True,
                           use_transpose_convolution=False)



        testGene = testGenerator(testSetX, inputSize=inputSize, inputChannels=3)

        NTest = len(testSetX)
        testSteps = np.ceil(NTest / batchSize)
        results = model.predict(testGene, verbose=1,batch_size=batchSize,steps=testSteps)


        if not os.path.exists(resultsPath):
            os.makedirs(resultsPath)
        saveResults(testSetX, results, resultsPath)

        #plt.subplot(2, 2, 1)
        #plt.plot(model.history['accuracy'])
        #plt.plot(model.history['val_accuracy'])
        #plt.title('Model accuracy')
        #plt.ylabel('Accuracy')
        #plt.xlabel('Epoch')
        #plt.legend(['Train', 'Test'], loc='lower right')
#
        ## Plot training & validation loss values
        #plt.subplot(2, 2, 2)
        #plt.plot(model.history['loss'])
        #plt.plot(model.history['val_loss'])
        #plt.title('Model loss')
        #plt.ylabel('Loss')
        #plt.xlabel('Epoch')
        #plt.legend(['Train', 'Test'], loc='upper right')

    # plt.subplot(2, 2, 3)
    # plt.plot(moving_average(batch_history.batch_accuracies, 5))
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Batch')
    # plt.legend(['Train'], loc='lower right')
    #

    # # Plot training & validation loss values
    # plt.subplot(2, 2, 4)
    # plt.plot(moving_average(batch_history.batch_losses, 5))
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Batch')
    # plt.legend(['Train'], loc='upper right')

        #plt.show()


if __name__ == '__main__':
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    main()