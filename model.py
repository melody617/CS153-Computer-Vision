import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# creates baseline CNN that we will use in a lot of our methods
def create_cnn():
    input_img = Input(shape=(96,96,3)) # size of each image in Patch Camelyon

    # Now we define the layers of the convolutional network: three blocks of two convolutional layers and a max-pool layer.
    x = Conv2D(16, (3, 3), padding='valid', activation='relu')(input_img)
    x = Conv2D(16, (3, 3), padding='valid', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    # flatten the output from a 4D to a 2D tensor to be able to use fully-connected (dense) layers for the final
    # classification part
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.2)(x) #dropout for regularization
    x = Dense(128, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    predictions = Dense(2, activation='softmax')(x) #softmax to obtain class likelihoods (return 0 or 1)

    # Now we define the inputs/outputs of the model and setup the optimizer. In this case we use regular stochastic gradient
    # descent with Nesterov momentum. The loss we use is cross-entropy and we would like to output accuracy as an additional metric.
    model = Model(inputs=input_img, outputs=predictions)
    sgd_opt = SGD(lr=0.01, momentum=0.9, decay=0.0, clipnorm=1.0, nesterov=True)
    model.compile(optimizer=sgd_opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    return model