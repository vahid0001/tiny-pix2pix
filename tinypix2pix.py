import keras
import numpy as np
import os

class TinyPix2Pix():
    def __init__(self, input_shape=(32, 32, 3), model_dir='models'):
        self.input_shape = input_shape
        self.model_dir = model_dir

    def define_unet(self):
        # UNet
        inputs = keras.layers.Input(self.input_shape)

        conv1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = keras.layers.Dropout(0.5)(conv4)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = keras.layers.Dropout(0.5)(conv5)

        up6 = keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(drop5))
        merge6 = keras.layers.concatenate([drop4, up6], axis=3)
        conv6 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = keras.layers.concatenate([conv3, up7], axis=3)
        conv7 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = keras.layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = keras.layers.concatenate([conv2, up8], axis=3)
        conv8 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = keras.layers.Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = keras.layers.concatenate([conv1, up9], axis=3)
        conv9 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = keras.layers.Conv2D(3, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)

        self.unet = keras.Model(input=inputs, output=conv9)


    def define_patchgan(self):
        # PatchNet
        source_image = keras.layers.Input(shape=self.input_shape)
        target_image = keras.layers.Input(shape=self.input_shape)

        merged = keras.layers.Concatenate()([source_image, target_image])

        x = keras.layers.Conv2D(64, 3, strides=2)(merged)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(128, 3)(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(256, 3)(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(512, 3)(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.25)(x)

        x = keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

        self.patchgan = keras.Model(input=[source_image, target_image], output=x)
        self.patchgan.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy')


    def define_tinypix2pix(self):
        self.define_unet()
        self.define_patchgan()

        self.patchgan.trainable = False

        input_source = keras.layers.Input(shape=self.input_shape)
        unet_output = self.unet(input_source)

        patchgan_output = self.patchgan([input_source, unet_output])

        self.tinypix2pix = keras.Model(input_source, [patchgan_output, unet_output])
        optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        self.tinypix2pix.compile(loss=['binary_crossentropy', 'mae'], optimizer=optimizer, loss_weights=[1, 100])


    def real_samples(self, dataset, batchsize):
        ix = np.random.randint(0, dataset.shape[0], batchsize)
        x_realA, x_realB = dataset[ix], dataset[ix]
        # 'real' class labels == 1
        y_real = np.ones((batchsize,) + self.patchgan.layers[-1].output_shape[1:])

        return [x_realA, x_realB], y_real


    def fake_samples(self, x_real):
        x_fake = self.unet.predict(x_real)
        # 'fake' class labels == 0
        y_fake = np.zeros((len(x_fake),) + self.patchgan.layers[-1].output_shape[1:])

        return x_fake, y_fake


    # train tinypix2pix model
    def fit(self, dataset, epochs, batchsize):
        # number of batches per training epoch
        batches = int(len(dataset) / batchsize)
        # number of training steps
        steps = batches * epochs
        # make directory for saving models
        os.makedirs(self.model_dir, exist_ok=True)

        self.define_tinypix2pix()

        # training loop
        for i in range(steps):
            # select a batch of real samples
            [x_realA, x_realB], y_real = self.real_samples(dataset, batchsize)
            # generate a batch of fake samples
            x_fakeB, y_fake = self.fake_samples(x_realA)
            # update discriminator for real samples
            d_loss1 = self.patchgan.train_on_batch([x_realA, x_realB], y_real)
            # update discriminator for generated samples
            d_loss2 = self.patchgan.train_on_batch([x_realA, x_fakeB], y_fake)
            # update the generator
            g_loss, _, _ = self.tinypix2pix.train_on_batch(x_realA, [y_real, x_realB])
            print('>%d/%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, steps, d_loss1, d_loss2, g_loss))
            # save the model each epoch
            if i % len(dataset) == 0:
                self.patchgan.save(self.model_dir + '/unet_' + str(i) + '.h5')
                self.patchgan.save(self.model_dir + '/patchgan_' + str(i) + '.h5')
                self.tinypix2pix.save(self.model_dir + '/tinypix2pix_' + str(i) + '.h5')


def main():
    # prepare data
    (x_train, _), (_, _) = keras.datasets.cifar10.load_data()
    x_train = x_train / 127.5 - 1
    # train model
    model = TinyPix2Pix(input_shape=(32, 32, 3))
    model.fit(x_train, epochs=1, batchsize=1)

if __name__ == '__main__':
    main()
