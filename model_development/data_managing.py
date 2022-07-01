import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
TRAIN_PATH = 'model_development/data/train'
TEST_PATH = 'model_development/data/test'

def print_train_dir():
    for expression in os.listdir('model_development/data/train'):
        print(str(len(os.listdir(f'data/train/{expression}'))) +f' {expression} images')

def create_train_vali_gens(batch_size=64):
    img_size = (48,48)
    datagen_train = ImageDataGenerator(horizontal_flip=True)
    train_generator = datagen_train.flow_from_directory(TRAIN_PATH,
                                                        target_size=img_size,
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    datagen_validation = ImageDataGenerator(horizontal_flip=True)
    validation_generator = datagen_validation.flow_from_directory(TEST_PATH,
                                                        target_size=img_size,
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=False)
    return (train_generator, validation_generator)

def main():
    print('Tensorflow version: ', tf.__version__)
    train_gen, vali_gen = create_train_vali_gens()
    print(next(vali_gen)[:,13,13,:])

if __name__ == '__main__':
    main()