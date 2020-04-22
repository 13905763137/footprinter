from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def generator(BS, data_path, npix):
  '''
  Image Data Generator from Directory;
  Given the path to the train, validation, test 
  directories located in data_path 
  preprocesses the images and the 
  target masks in batch sizes of BS.
  '''  
  train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 45,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True, 
        fill_mode = "nearest")
        
  train_maskgen = ImageDataGenerator(
        rescale=1.,
        rotation_range = 45,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode = "nearest")
  
  val_datagen = ImageDataGenerator(rescale=1./255)
  val_maskgen = ImageDataGenerator(rescale=1.)

  test_datagen = ImageDataGenerator(rescale=1./255)
  test_maskgen = ImageDataGenerator(rescale=1.)
  
  train_image_generator = train_datagen.flow_from_directory(
                        os.path.join(data_path, 'train/images/pre/'),
                        class_mode=None, 
                        target_size = (npix, npix),
                        seed = 12345,
                        batch_size = BS)

  train_mask_generator = train_maskgen.flow_from_directory(
                       os.path.join(data_path, 'train/targets/pre/'),
                       class_mode=None,
                       target_size = (npix, npix),
                       color_mode = "grayscale",
                       seed = 12345,
                       batch_size = BS)

  val_image_generator = val_datagen.flow_from_directory(
                      os.path.join(data_path, 'test/images/pre/'),
                      class_mode=None,
                      target_size = (npix, npix),
                      seed = 123,
                      batch_size = BS)


  val_mask_generator = val_maskgen.flow_from_directory(
                     os.path.join(data_path, 'test/targets/pre2/'),
                     class_mode=None,
                     target_size = (npix, npix),
                     color_mode = "grayscale",
                     seed = 123,
                     batch_size = BS)

  test_image_generator = val_datagen.flow_from_directory(
                      os.path.join(data_path, 'hold/images/pre/'),
                      class_mode=None,
                      target_size = (npix, npix),
                      seed = 123,
                      batch_size = BS)


  test_mask_generator = val_maskgen.flow_from_directory(
                     os.path.join(data_path, 'hold/targets/pre2/'),
                     class_mode=None,
                     target_size = (npix, npix),
                     color_mode = "grayscale",
                     seed = 123,
                     batch_size = BS)
  train_generator = zip(train_image_generator, train_mask_generator)
  valid_generator = zip(val_image_generator, val_mask_generator)
  test_generator = zip(test_image_generator, test_mask_generator)

  return train_generator, valid_generator, test_generator
