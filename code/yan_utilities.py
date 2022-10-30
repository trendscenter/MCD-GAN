import scipy.io as scio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import mat73
import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,auc

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def change_mat_to_npy(mat_file_names):
    count = 1
    for i in mat_file_names:
        print(i)
        temp = mat73.loadmat(i)
        x = temp['img']
        x = x*2.0-1
        np.save(i[:-4],x)
        count = count+1
        print(str(count)+'...'+str(np.shape(x)))

def read_npy_file(filename, label):
    '''
    Example: data = tf.py_function(read_npy_file,[npy_file_names[0]],[tf.float32])

    :param filename:
    :return:
    '''
    # print(filename)
    data = np.load(filename.numpy().decode())
    data = data[12: -13, 8: -9, 12: -13]
    data = np.expand_dims(data, axis=3)
    return data.astype(np.float32), label

def read_image(filename, label):
    '''
    Example: read_image(npy_file_names[0])

    :param filename:
    :return:
    '''
    image_, label = tf.py_function(read_npy_file, [filename, label], [tf.float32, tf.float32])
    print(tf.shape(image_))
    return image_, label

def label_to_onehot(label):
    '''change the 2 class label (label should be 0 or 1) to one hot'''
    # one_hot_label = np.zeros([len(label),2])
    # for index in range(len(label)):
    #     one_hot_label[index,int(label[index])] = 1
    unique_categories = np.unique(label)
    num_categories = len(unique_categories)
    one_hot_label = np.zeros([len(label),num_categories])
    for i in range(len(label)):
        for which_column in range(num_categories):
            if unique_categories[which_column] == label[i]:
                break
        one_hot_label[i, which_column] = 1
    return one_hot_label

def data_preprocessing_and_domain_selection(data_name,domain_1_index,domain_2_index,output_label_one_hot=True, feature_type='vector'):
    '''

    Parameters
    ----------
    data_name
    domain_1_index
    domain_2_index
    output_label_one_hot
    feature_type

    Returns
    -------
    e.g.,

    data_name = 'result/Origin_MRI3D_data_category_domain.mat'
    domain_1_index=1
    domain_2_index=3
    data_name_domain1, data_name_domain2, category_label_domain1_one_hot, category_label_domain2_one_hot = data_preprocessing_and_domain_selection(data_name,domain_1_index,domain_2_index,output_label_one_hot=True, feature_type='MRI3D')
    data_name_domain1 = pd.DataFrame(data_name_domain1)

    data_name_domain1 = 'data/T1_data/' + pd.DataFrame(data_name_domain1) + '.npy'
    data_name_domain2 = 'data/T1_data/' + pd.DataFrame(data_name_domain2) + '.npy'
    '''

    if feature_type in ['vector']:
        try:
            dataset = mat73.loadmat(data_name)
        except:
            dataset = scio.loadmat(data_name)

        data = dataset['data']
        domain_label = dataset['domain_label']
        category_label = dataset['category_label']

        if len(np.shape(category_label)) == 1:
            category_label = np.expand_dims(category_label,axis=1)
            domain_label = np.expand_dims(domain_label,axis=1)

        data_category = np.concatenate([data, category_label], axis=1)

        data_category_domain1 = data_category[np.where(domain_label==domain_1_index)[0],:] # domain1 and category label
        data_category_domain2 = data_category[np.where(domain_label==domain_2_index)[0],:] # domain2 data and category label

        data_domain1 = data_category_domain1[:,:-1]
        data_domain2 = data_category_domain2[:,:-1]

        category_domain1 = data_category_domain1[:,-1]  ## domain1 label
        category_domain2 = data_category_domain2[:,-1] ## domain2 label

        category_label_domain1_one_hot = label_to_onehot(category_domain1)
        category_label_domain2_one_hot = label_to_onehot(category_domain2)

        if output_label_one_hot is True:
            return data_domain1, data_domain2, category_label_domain1_one_hot, category_label_domain2_one_hot
        else:
            return data_domain1, data_domain2, category_domain1, category_domain2

    if feature_type in ['MRI3D']:
        dataset = mat73.loadmat(data_name)
        data_name = np.array(dataset['data_name'])

        domain_label = dataset['domain_label']
        category_label = dataset['category_label']

        category_label = np.expand_dims(category_label, axis=1)
        domain_label = np.expand_dims(domain_label, axis=1)

        data_name_domain1 = data_name[np.where(domain_label == domain_1_index)[0]]
        data_name_domain2 = data_name[np.where(domain_label == domain_2_index)[0]]

        category_domain1 = category_label[np.where(domain_label == domain_1_index)[0],:]  # domain1 and category label
        category_domain2 = category_label[np.where(domain_label == domain_2_index)[0],:]  # domain2 data and category label

        category_label_domain1_one_hot = label_to_onehot(category_domain1)
        category_label_domain2_one_hot = label_to_onehot(category_domain2)

        if output_label_one_hot is True:
            return data_name_domain1, data_name_domain2, category_label_domain1_one_hot, category_label_domain2_one_hot
        else:
            return data_name_domain1, data_name_domain2, category_domain1, category_domain2

def preprocess_vector_train(vector,label):
    return vector,label


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    """Downsamples an input.

    Conv2D => Batchnorm => LeakyRelu

    Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer

    Returns:
    Downsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    # result.add(
    #     tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
    #                            kernel_initializer=initializer, use_bias=False))
    result.add(
      tf.keras.layers.Conv3D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """Upsamples an input.

    Conv3DTranspose => Batchnorm => Dropout => Relu

    Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

    Returns:
    Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    # result.add(
    #     tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
    #                                     padding='same',
    #                                     kernel_initializer=initializer,
    #                                     use_bias=False))
    result.add(
      tf.keras.layers.Conv3DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def unet_generator(output_channels, norm_type='batchnorm'):
    """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

    Args:

    Returns:
    Generator model
    """

    down_stack = [
      downsample(32, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
      downsample(64, 4, norm_type),  # (bs, 64, 64, 128)
      downsample(128, 4, norm_type),  # (bs, 32, 32, 256)
      downsample(128, 4, norm_type),  # (bs, 16, 16, 512)
      downsample(128, 4, norm_type),  # (bs, 8, 8, 512)
      # downsample(128, 4, norm_type),  # (bs, 2, 2, 512)
      # downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
    ]

    up_stack = [
      upsample(128, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
      # upsample(128, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
      upsample(128, 4, norm_type),  # (bs, 16, 16, 1024)
      upsample(128, 4, norm_type),  # (bs, 32, 32, 512)
      upsample(64, 4, norm_type),  # (bs, 64, 64, 256)
      upsample(32, 4, norm_type),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv3DTranspose(
      output_channels, 4, strides=2,
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    # inputs = tf.keras.layers.Input(shape=[None, None, 3])
    inputs = tf.keras.layers.Input(shape=[None, None, None, 1])

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
def discriminator(norm_type='batchnorm', target=True):
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.

    Returns:
    Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    # inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    inp = tf.keras.layers.Input(shape=[None, None, None, 1], name='input_image')

    x = inp

    if target:
        # tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
        tar = tf.keras.layers.Input(shape=[None, None, None, 1], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv3D(
      512, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv3D(
      1, 4, strides=1,
      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1

def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

##############################################################
def classifier_loss(predicted,label):
    return loss_obj(predicted,label) ### Remeber changing the variable to tf.variables
##############################################################

def generate_images(model, test_input,save_path=None):
    '''
    example: generate_images(generator_g_scan, sample_GE, save_path='cycle_gan_simple_'+str(epoch)+'.png')
    :param model:
    :param test_input:
    :param save_path:
    :return:
    '''

    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input.numpy(), prediction.numpy()]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)


def get_classifier(classifier_name, input_vector_size=68, ):
    inputs = layers.Input(shape=input_vector_size) # The input is a vector.

    if classifier_name in ['DemoDNN']:
        x = layers.Dense(32)(inputs)
        x = layers.LeakyReLU(0.4)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(32)(x)
        x = layers.LeakyReLU(0.4)(x)

        x = layers.Dense(2)(x)
        x = layers.LeakyReLU(0.4)(x)

        x = layers.Softmax()(x)
        model = keras.models.Model(inputs=inputs, outputs=x)  # x is the softmax output
        return model

    if classifier_name in ['DNN']:

        x = layers.Dense(128)(inputs)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.2)(x)

        # x = layers.Dense(256)(x)
        # x = layers.LeakyReLU(0.2)(x)
        # x = layers.Dropout(0.2)(x)

        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(2)(x)
        x = layers.Softmax()(x)
        model = keras.Model(inputs=inputs, outputs=x)  # x is the softmax output

    if classifier_name in ['3DCNN']:
        x = layers.Conv3D(filters=32, kernel_size=3, activation='relu')(inputs)
        x = layers.MaxPooling3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=32, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(units=256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(2, activation='sigmoid')(x)
        x = layers.Softmax()(x)


        model = keras.Model(inputs=inputs, outputs=x, name="3dcnn")

    return model

def get_generator(generator_name='DNN', input_vector_size=68):

    inputs = layers.Input(shape=input_vector_size) # The input is a vector.

    if generator_name in ['DemoDNN']:
        x = layers.Dense(32)(inputs)
        x = layers.Activation(tf.nn.leaky_relu)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(32)(x)
        x = layers.Activation(tf.nn.leaky_relu)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(input_vector_size)(x)

        model = keras.models.Model(inputs, x)

    if generator_name in ['DNN']:

        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(128)(x)
        x = layers.Activation(tf.nn.leaky_relu)(x)
        # x = layers.Dropout(0.2)(x)

        # x = layers.Dense(256)(x)
        # x = layers.Activation(tf.nn.leaky_relu)(x)
        # x = layers.Dropout(0.2)(x)

        x = layers.Dense(128)(x)
        x = layers.Activation(tf.nn.leaky_relu)(x)

        x = layers.Dense(input_vector_size)(x)
        x = layers.Activation(tf.nn.tanh)(x)

        model = keras.models.Model(inputs, x)

    if generator_name in ['3DCNN']:
        norm_type = 'batchnorm'  # 'batchnorm' or 'instancenorm'.
        output_channels = 1

        down_stack = [
            downsample(32, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
            downsample(64, 4, norm_type),  # (bs, 64, 64, 128)
            downsample(128, 4, norm_type),  # (bs, 32, 32, 256)
            downsample(128, 4, norm_type),  # (bs, 16, 16, 512)
            downsample(128, 4, norm_type),  # (bs, 8, 8, 512)
            # downsample(128, 4, norm_type),  # (bs, 2, 2, 512)
            # downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            upsample(128, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
            # upsample(128, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
            upsample(128, 4, norm_type),  # (bs, 16, 16, 1024)
            upsample(128, 4, norm_type),  # (bs, 32, 32, 512)
            upsample(64, 4, norm_type),  # (bs, 64, 64, 256)
            upsample(32, 4, norm_type),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv3DTranspose(
            output_channels, 4, strides=2,
            padding='same', kernel_initializer=initializer,
            activation='tanh')  # (bs, 256, 256, 3)

        concat = tf.keras.layers.Concatenate()

        # inputs = tf.keras.layers.Input(shape=[None, None, 3])
        inputs = tf.keras.layers.Input(shape=[None, None, None, 1])

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])

        x = last(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

def get_discriminator(discriminator_name='DNN', input_vector_size=68):
    inputs = layers.Input(shape=input_vector_size) # The input is a vector.

    if discriminator_name in ['DemoDNN']:
        x = layers.Dense(32)(inputs)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(32)(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Dense(1)(x)

        model = keras.models.Model(inputs, x)

    if discriminator_name in ['DNN']:

        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(128)(x)

        x = layers.LeakyReLU(0.2)(x)
        # x = layers.Dropout(0.2)(x)

        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Dense(1)(x)
        x = layers.Activation('sigmoid')(x)

        model = keras.models.Model(inputs, x)

    if discriminator_name in ['3DCNN']:

        norm_type = 'batchnorm'  # 'batchnorm' or 'instancenorm'.
        target = False # target: Bool, indicating whether target image is an input or not.
        initializer = tf.random_normal_initializer(0., 0.02)

        # inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
        inp = tf.keras.layers.Input(shape=[None, None, None, 1], name='input_image')

        x = inp

        if target:
            # tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
            tar = tf.keras.layers.Input(shape=[None, None, None, 1], name='target_image')

            x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

        down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
        down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
        down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv3D(
            512, 4, strides=1, kernel_initializer=initializer,
            use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        if norm_type.lower() == 'batchnorm':
            norm1 = tf.keras.layers.BatchNormalization()(conv)
        elif norm_type.lower() == 'instancenorm':
            norm1 = InstanceNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

        zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv3D(
            1, 4, strides=1,
            kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        if target:
            model = tf.keras.Model(inputs=[inp, tar], outputs=last)
        else:
            model = tf.keras.Model(inputs=inp, outputs=last)

    return model

