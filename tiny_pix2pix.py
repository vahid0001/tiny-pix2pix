import keras
from keras.layers import *
from keras import backend as keras
from keras.optimizers import Adam
from keras.models import Model
from keras.datasets import cifar10
import numpy as np


def define_unet():
    # UNet
    inputs = Input((32, 32, 3))

    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)

    unet = Model(input=inputs, output=conv9)
    
    return unet


def define_patchnet():
    # PatchNet
    source_image = Input(shape=(32, 32, 3))
    target_image = Input(shape=(32, 32, 3))

    merged = Concatenate()([source_image, target_image])

    x = Conv2D(64, 3, strides=2)(merged)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, 3)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, 3)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(512, 3)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    patchnet = Model(input=[source_image, target_image], output=x)
    patchnet.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')
    
    return patchnet



def define_pix2pix(generator, discriminator, image_shape):
    discriminator.trainable = False
    
    input_source = Input(shape=image_shape)
    generator_output = generator(input_source)

    discriminator_output = discriminator([input_source, generator_output])

    pix2pix = Model(input_source, [discriminator_output, generator_output])
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    pix2pix.compile(loss=['binary_crossentropy', 'mae'], optimizer=optimizer, loss_weights=[1,100])
    
    return pix2pix



def real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X1, X2 = dataset[ix], dataset[ix]
    # 'real' class labels == 1
    y = np.ones((n_samples, 9, 9, 1))
    
    return [X1, X2], y



def fake_samples(generator, samples):
    X = generator.predict(samples)
    # 'fake' class labels == 0
    y = np.zeros((len(X), 9, 9, 1))
    
    return X, y



# train tiny-pix2pix model
def train(discriminator, generator, pix2pix, dataset, epochs=20, batch=1):
    # number of batches per training epoch
    batches = int(len(dataset) / batch)
    # number of training steps
    steps = batches * epochs
    # training loop
    for i in range(steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = real_samples(dataset, batch)
        # generate a batch of fake samples
        X_fakeB, y_fake = fake_samples(generator, X_realA)
        # update discriminator for real samples
        d_loss1 = discriminator.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = discriminator.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = pix2pix.train_on_batch(X_realA, [y_real, X_realB])
        # save the model each epoch
        if i % len(dataset) == 0:
            print('>%d/%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, steps, d_loss1, d_loss2, g_loss))
            generator.save('unet' + str(i) +'.h5')
            discriminator.save('patchnet' + str(i) +'.h5')
            pix2pix.save('pix2pix' + str(i) +'.h5')
            
            
def main():
    (x_train, _), (_, _) = cifar10.load_data()

    # normalize data
    x_train = x_train / 127.5 - 1
    
    # define Pix2Pix
    unet = define_unet()
    patchnet = define_patchnet()
    
    pix2pix = define_pix2pix(unet, patchnet, (32, 32, 3))

    # training
    train(patchnet, unet, pix2pix, dataset=x_train)
    
    
    
if __name__ == '__main__':
    main()
