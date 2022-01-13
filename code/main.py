import os
import sys
import random
import argparse
import numpy as np
import tensorflow as tf
import unicodedata

from vocab import Vocab
from cnn import RepVGG
from augment import ImageAugmentor
from attnocr import AttentionOCR

from tensorflow.keras import Sequential, Model, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.losses import Loss, CategoricalCrossentropy
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD


# Create a tf.data.Dataset for the 3 sets
def createDataset(data):
    images, labels = data[:, 0], data[:, 1]
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    #dataset = dataset.map(image_augment, num_parallel_calls=4)          # for data augmentation
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

# A function to load and preprocess the images
def parse_function(filename, label):
    image_string = tf.io.read_file(data_image_dir+filename)
    
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    # Resize the image to a fix size to feed in model
    image = tf.image.resize(image, [64, 128], method='nearest')
    
    # Convert label from text to one hot vector
    label = vocab.tf_encode(label)
    #tf.py_function(show_content, [label], tf.float32)
    label = tf.one_hot(tf.cast(label, tf.int32), len(vocab))
    #tf.py_function(show_content, [label], tf.float32)
    
    return image, label

def image_augment(image, label):
    transform = ImageAugmentor()
    image, label = transform.augment_tf(image, label)
    
    return image, label

# Function to convert repvgg to inference architecture
def repvgg_convert(model:tf.Module):
    num_blocks = [2, 4, 14, 1]  # [2, 4, 14, 1] for RepVGG-A or [4, 6, 16, 1] for RepVGG-B
    width_multipliers = [0.75, 0.75, 0.75, 2]
    deploy = RepVGG(num_blocks, width_multipliers, dropout=0.3, deploy=True)
    deploy.build(input_shape=(None, 64, 128, 3))
    for layer, deploy_layer in zip(model.layers, deploy.layers):
        if hasattr(layer, "convert_inference"):
            kernel, bias = layer.convert_inference()
            deploy_layer.reparam.layers[1].set_weights([kernel, bias])
        elif isinstance(layer, Sequential):
            for sublayer, deploy_sublayer in zip(layer.layers, deploy_layer.layers):
                if hasattr(sublayer, "convert_inference"):
                    kernel, bias = sublayer.convert_inference()
                    deploy_sublayer.reparam.layers[1].set_weights([kernel, bias])
        elif isinstance(layer, tf.keras.layers.Dense):
            assert isinstance(deploy_layer, tf.keras.layers.Dense)
            weights = layer.get_weights()
            deploy_layer.set_weights(weights)

    return deploy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--trainimage", type=str, help='a path to the image data')
    parser.add_argument("-n", "--filename", type=str, help='a path to the txt data file')
    parser.add_argument("-o", "--output", type=str, help='a path to the output file')

    args = parser.parse_args()

    data_image_dir = args.trainimage
    data_txt = args.filename
    save_path = args.output

    # Check if the input for command lines is correct
    if not os.path.isdir(data_image_dir):
        print("Directory does not exist, please check again.")
        sys.exit(1)
    if not os.path.isfile(data_txt):
        print("File does not exist, please check again.")
        sys.exit(1)
    # create a file name for the output file if not specified
    if not os.path.isdir(save_path):
        print("Directory does not exist, please check again.")
        sys.exit(1)

    # variables for handling images
    batch_size = 128                # batch for feeding data
    buffer_size = batch_size*6      # buffer size for shuffering
    vocab = Vocab()

    # preprocess input images
    with tf.device('/cpu:0'):
        data_file = open(data_txt, encoding="utf-8")
        data = np.genfromtxt(data_file, delimiter="\t", dtype="str")
        for line in data:
            line[1] = unicodedata.normalize('NFKC', line[1])

        # Shuffle and split to 3 sets
        random.shuffle(data)
        num_train = round(len(data)*0.6)
        num_val = round(len(data)*0.2)
        train = data[:num_train]
        val = data[num_train:num_train+num_val]
        test = data[num_train+num_val:]
        
        train_set = createDataset(train)
        print("Train set completed!")
        val_set = createDataset(val)
        print("Validation set completed!")
        test_set = createDataset(test)
        print("Test set completed!")

    #print(train_set)
    # parameters for Attention OCR model
    num_blocks = [2, 4, 14, 1]  # [2, 4, 14, 1] for RepVGG-A or [4, 6, 16, 1] for RepVGG-B
    width_multipliers = [0.75, 0.75, 0.75, 2] 

    # parameters for training process
    lnr = 0.001
    epochs = 2

    # Create model and define inputs
    model = AttentionOCR(num_blocks, width_multipliers, 128, dropout=0.2, vocab=vocab)
    input_img = Input(shape=(64, 128, 3))
    input_label = Input(shape=(vocab.max_seq_length, len(vocab)))
    model([input_img, input_label])

    # Define optimizers, loss function and metrics
    #adam = Adam(learning_rate=lnr, amsgrad=True)
    sgd = SGD(learning_rate=lnr, momentum=0.2)
            
    cce = CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    sc_accu = CategoricalAccuracy(name='acc')

    # Compile model
    model.compile(optimizer=sgd, loss=cce, metrics=[sc_accu], run_eagerly=True)
    print("Compile is done!")

    # Show summary
    model.summary()

    # implement callbacks for early stopping
    earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", 
                                            patience=5, restore_best_weights = True)

    # Train model woth train_set and validate with val_set
    history = model.fit(train_set, epochs=epochs, validation_data=val_set, 
                        shuffle=True, verbose=1, workers=6, use_multiprocessing=True, 
                        callbacks=[earlystopping])

    model.evaluate()

    # convert model to deploy
    deploy_model = model
    for layer in deploy_model.layers:
        if hasattr(layer, "make_stage"):
            repvgg_deploy = repvgg_convert(layer)
            deploy_model.cnn = repvgg_deploy

    deploy_model.summary()
    # Save model to output folder
    deploy_model.save(save_path)
