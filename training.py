"""
#################################
 Training phase after demonstration: This module uses Keras and Tensor flow to train the image classification problem
 for the labeling fire and non-fire data based on the aerial images.
 Training and Validation Data: Item 7 on https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
 Keras version: 2.4.0
 Tensorflow Version: 2.3.0
 GPU: Nvidia RTX 2080 Ti
 OS: Ubuntu 18.04
#################################
"""

#########################################################
# import libraries

import os.path
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


from config import new_size
from plotdata import plot_training
from config import Config_classification
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

#########################################################
# Global parameters and definition

data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

image_size = (new_size.get('width'), new_size.get('height'))
batch_size = Config_classification.get('batch_size')
save_model_flag = Config_classification.get('Save_Model')
epochs = Config_classification.get('Epochs')

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.Accuracy(name='accuracy'),
    keras.metrics.BinaryAccuracy(name='bin_accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc')
]


#########################################################
# Function definition

def train_keras():
    """
    This function train a DNN model based on Keras and Tensorflow as a backend. At first, the directory of Fire and
    Non_Fire images should be defined for the model, then the model is defined, compiled and fitted over the training
    and validation set. At the end, the models is saved based on the *.h5 parameters and weights. Training accuracy and
    loss are demonstrated at the end of this function.
    :return: None, Save the trained model and plot accuracy and loss on train and validation dataset.
    """
    # This model is implemented based on the guide in Keras (Xception network)
    # https://keras.io/examples/vision/image_classification_from_scratch/
    print(" --------- Training --------- ")

    dir_fire = 'frames/Training/Fire/'
    dir_no_fire = 'frames/Training/No_Fire/'

    # 0 is Fire and 1 is NO_Fire
    fire = len([name for name in os.listdir(dir_fire) if os.path.isfile(os.path.join(dir_fire, name))])
    no_fire = len([name for name in os.listdir(dir_no_fire) if os.path.isfile(os.path.join(dir_no_fire, name))])
    total = fire + no_fire
    weight_for_fire = (1 / fire) * total / 2.0
    weight_for_no_fire = (1 / no_fire) * total / 2.0
    # class_weight = {0: weight_for_fire, 1: weight_for_no_fire}

    print("Weight for class fire : {:.2f}".format(weight_for_fire))
    print("Weight for class No_fire : {:.2f}".format(weight_for_no_fire))

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Training", validation_split=0.2, subset="training", seed=1337, image_size=image_size,
        batch_size=batch_size, shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Training", validation_split=0.2, subset="validation", seed=1337, image_size=image_size,
        batch_size=batch_size, shuffle=True
    )

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            _ = plt.subplot(3, 3, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")

    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            _ = plt.subplot(3, 3, i+1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    # Model 1:
    # model = make_model_ResNet50()

    # Model 2:
    model = make_model_InceptionV3()




    keras.utils.plot_model(model, show_shapes=True)

    callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"), ]
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"], )
    res_fire = model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, batch_size=batch_size)

    layers_len = len(model.layers)

    if save_model_flag:
        file_model_fire = 'Output/Models/model_fire_resnet_weighted_40_no_metric_simple'
        model.save(file_model_fire)
    #if Config_classification.get('TrainingPlot'):
    #   plot_training(res_fire, 'KerasModel', layers_len)

    # Prediction on one sample frame from the test set
    img = keras.preprocessing.image.load_img(
        "frames/Training/Fire/resized_frame0.jpg", target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    print("This image is %.2f percent Fire and %.2f percent No Fire." % (100 * (1 - score), 100 * score))
def make_model_InceptionV3():
    base_model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.models.Model(base_model.input, x)



def make_model_ResNet50():
    base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    for layer in base_model.layers:
        layer.trainable = False
    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    base_model.add(Dense(1, activation='sigmoid'))

    return base_model

