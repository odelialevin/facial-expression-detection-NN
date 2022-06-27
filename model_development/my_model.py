from tensorflow.keras.layers import Dense, Dropout,Flatten, Conv2D,BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from IPython.display import Image



def create_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), input_shape=(48, 48, 1) ,padding='valid'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(7, activation='softmax'))

    adam = Adam(learning_rate=0.0002)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    model = create_model()
    model.summary()
    plot_model(model, to_file='model7.png', show_shapes=True, show_layer_names=True)
    Image('model.png',width=400, height=200)

if __name__ == '__main__':
    main()