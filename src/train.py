import os
import numpy as np 
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from datagen import generator
from mobile import unet_model
from baseline_model import baseline_unet_model

def train_cnn(args):
    '''
    train the Unet-Style model with the 
    batch data from the data generator'''
    #height/width of inputs and output masks
    npix = args["input_shape"]
    # learning rate
    lr = args["learning_rate"]
    # batch size
    BS = args["batch_size"]
    # number of epochs
    NEPOCHS = args["NEPOCHS"]
    # path to the data directory
    data_path = args["data_path"]
    # path for saving the model
    model_path = args["model_path"]
    #type of model to use
    model_type = args["model_type"]
    # height, width, nchannels of input images
    input_shape = (npix, npix, 3)
    # call the train and val batch data generator
    train_gen, val_gen, _ = generator(BS, data_path, npix)
    # call the model
    if model_type == "mobilenet":
       model = unet_model(input_shape)
    elif model_type == "baseline":
       model = baseline_unet_model(input_shape)
    else:
       raise ValueError
    # compile the imported model
    model.compile(optimizer = Adam(lr = lr), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])
    # create checkpoints for monitoring the validation accuracy
    checkpoint = ModelCheckpoint(model_path, 
                                 monitor='val_accuracy', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')
    # keep track of changes in the metrics after each epoch
    csv_logger = CSVLogger(model_path+'log_lr_{:}_BS_{:}_model_{:}.out'.format(lr, BS, model_type), append=True, separator=',')
    # prevent overfitting by early stopping if validation accuracy goes up
    earlystopping = EarlyStopping(monitor = 'val_accuracy', 
                                  verbose = 1,
                                  min_delta = 0.1, 
                                  patience = 3, 
                                  mode = 'max')
    # put all the callbacks together
    callbacks_list = [checkpoint, csv_logger, earlystopping]
    # fit the model
    results = model.fit_generator(train_gen, 
                                  epochs = NEPOCHS, 
                                  steps_per_epoch = (2799//BS),
                                  validation_data = val_gen, 
                                  validation_steps = (933//BS),
                                  callbacks = callbacks_list)
    # save the model
    model.save(model_path+'Model_lr_{:}_BS_{:}_model_{:}.h5'.format(lr, BS, model_type))
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-npix",
                        "--input_shape", 
                        type = int,
                        required = True, 
                        help = "Height/widths of the input images")
    parser.add_argument("-lr",
                        "--learning_rate", 
                        type = float,
                        required = True, 
                        help = "Learning Rate")
    parser.add_argument("-BS", 
                        "--batch_size",
                        type = int,
                        required = True, 
                        help = "Number of instances in training batch")
    parser.add_argument("-epochs",
                        "--NEPOCHS",
                        type = int, 
                        required = True, 
                        help = "Number of training epochs")
    parser.add_argument("-DPATH",
                        "--data_path",
                        type = str,
                        required = True, 
                        help = "Data directory path")
    parser.add_argument("-MPATH",
                        "--model_path",
                        type = str,
                        required = True, 
                        help = "Path for saving the model")
    parser.add_argument("-model",
                        "--model_type",
                        type = str,
                        required = True, 
                        help = "Which model should I use? Baseline or MobileNet")
    # put all the arguements together
    args = vars(parser.parse_args())
    # start training
    train_cnn(args)
