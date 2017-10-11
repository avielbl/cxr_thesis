import numpy as np
from numpy.random import seed as npseed
npseed(1)
from keras.models import Model
from keras.models import load_model as keras_load_model
from keras.layers import Input, merge, MaxPooling2D, UpSampling2D, Dropout, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.optimizers import SGD, rmsprop, Adam
import keras.callbacks
from keras import backend as K
import keras.callbacks
from drawnow import drawnow
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.legend_handler import HandlerLine2D
import time

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code


def load_model(model_path, custom_objects = None):
    if custom_objects is None:
        return keras_load_model(model_path)
    elif custom_objects == 'dice_coef_loss':
        return keras_load_model(model_path, custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef})
    else:
        return keras_load_model(model_path, custom_objects=custom_objects)


def dice_coef(y_true, y_pred):
    '''
    Computes dice cooeficient based on Sørensen's formula with smooth factor to avoid devision by zero
    2 binary matrices with same shape are assumed

    :param y_true: first binary image
    :param y_pred: second binary image
    :return: dice value range from 0 to 1
    '''
    smooth = 1.
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def dice_coef_loss(y_true, y_pred):
    '''
    Simply inversing dice cooeficient so 0 is perfect match and 1 is no match to be served as a loss function

    :param y_true: first binary image
    :param y_pred: second binary image
    :return: inverse of dice coef
    '''
    return 1 - dice_coef(y_true, y_pred)


def get_unet(im_size, filters=64, filter_size=3, dropout_val=0.5, lrelu_alpha=0.25, loss_fun=dice_coef_loss, metrics=dice_coef, optim_fun=None, **kwargs):
    '''
    Generate a compiled customized U-Net model for segmentation tasks.
    Assumes square grayscale images.

    :param im_size: input image size in pixels
    :param filters: number of filters in the first layer that will be doubled along contraction path (default to 64)
    :param filter_size: size in pixels for all convolution kernels (default to 3)
    :param dropout_val: value for the 2 dropout layers (default to 0.5)
    :param lrelu_alpha: paramter for the leaky relu. set to 0 for a standard relu (default to 0.25)
    :param loss_fun:loss function name as string or function name (default to dice_coeff_loss)
    :param metrics: metric name as string or function name (default to dice_coeff)
    :param optim_fun: optimizer to be used (default to SGD)
    :param kwargs: parameters that will be transferred to SGD optimizier
    :return: compiled keras model object
    '''
    def contraction_block(in_layer, filters_mult=1):
        '''
        Internal function implementing a single contracting block of layers.
        The block is composed of conv->leakyRelu->conv->leakyRelu->maxPooling

        :param in_layer: tensor object input layer to the block
        :param filters_mult: scalar multiplication factor for the number of filters
        :return: tuple of (conv_layer, pool_layer)
        '''
        conv_layer = Conv2D(filters * filters_mult,
                            (filter_size, filter_size),
                            kernel_initializer='he_normal',
                            padding='same')(in_layer)
        conv_layer = LeakyReLU(lrelu_alpha)(conv_layer)
        conv_layer = Conv2D(filters * filters_mult,
                            (filter_size, filter_size),
                            kernel_initializer='he_normal',
                            padding='same')(conv_layer)
        conv_layer = LeakyReLU(lrelu_alpha)(conv_layer)
        pool_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer)
        return conv_layer, pool_layer

    def expansion_block(low_res_layer, high_res_layer, filters_mult):
        '''
        Internal function implementing a single expansion block of layers.
        The block is composed of merge of lowResLayer and HighResLayer->conv->leakyRelu->conv->leakyRelu

        :param low_res_layer: output layer in low resolution from contracting path
        :param high_res_layer: output layer in high res from previous expansion block
        :param filters_mult: scalar multiplication factor for the number of filters
        :return: last conv layer in the block
        '''
        up_layer = concatenate([UpSampling2D(size=(2, 2))(low_res_layer), high_res_layer], axis=1)
        conv_layer = Conv2D(filters * filters_mult,
                            (filter_size, filter_size),
                            kernel_initializer='he_normal',
                            padding='same')(up_layer)
        conv_layer = LeakyReLU(lrelu_alpha)(conv_layer)
        conv_layer = Conv2D(filters * filters_mult,
                            (filter_size, filter_size),
                            kernel_initializer='he_normal',
                            padding='same')(conv_layer)
        conv_layer = LeakyReLU(lrelu_alpha)(conv_layer)
        return conv_layer

    inputs = Input((1, im_size, im_size))
    conv1, pool1 = contraction_block(inputs, 1)
    conv2, pool2 = contraction_block(pool1, 2)
    conv3, pool3 = contraction_block(pool2, 4)
    conv4, pool4 = contraction_block(pool3, 8)
    conv5, pool5 = contraction_block(pool4, 16)

    conv5 = Dropout(dropout_val)(conv5)

    conv6 = expansion_block(conv5, conv4, 8)
    conv7 = expansion_block(conv6, conv3, 4)
    conv8 = expansion_block(conv7, conv2, 2)
    conv9 = expansion_block(conv8, conv1, 1)

    conv9 = Dropout(dropout_val)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    if optim_fun == None:
        optim_fun = SGD(**kwargs)
    model.compile(optimizer=optim_fun, loss=loss_fun, metrics=[metrics])
    return model


def print_model_to_file(model, file_name=None):
    """
    Utility function for the generation of model's architecture image saved to png file

    :param model: the keras model object to be plotted
    :param file_name: (optional) name of image file. Default to 'model_HH_MM_dd_mm_yy.png'
    :return: None
    """
    from keras.utils.vis_utils import plot_model
    import time
    if file_name == None:
        file_name = 'model_' + time.strftime("%H_%M_%d_%m_%Y") + '.png'
    plot_model(model, to_file=file_name, show_shapes=True)


def plot_first_layer(model):
    layer = model.layers[0]
    if 'input' in layer.name:
        layer = model.layers[1]
    filters = layer.get_weights()[0]
    nb_filters = filters.shape[3]
    filt_sz = filters.shape[0]
    rows_of_filts = np.ceil(np.sqrt(nb_filters)).astype(int)
    cols_of_filts = np.ceil(np.sqrt(nb_filters)).astype(int)
    pad_sz = 2
    rows = filt_sz * rows_of_filts + pad_sz * (rows_of_filts - 1)
    cols = filt_sz * cols_of_filts + pad_sz * (cols_of_filts - 1)
    out_img = np.zeros((rows, cols))
    cur_row = 0
    cur_col = 0
    for i in range(nb_filters):
        out_img[cur_row:cur_row+filt_sz, cur_col:cur_col+filt_sz] = np.squeeze(filters[:,:,:,i])
        cur_col += filt_sz + pad_sz
        if cur_col > cols:
            cur_col = 0
            cur_row += filt_sz + pad_sz

    fig, ax = plt.subplots()
    cax = ax.imshow(out_img, cmap='gray')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fig.colorbar(cax)

def optimize_ticks(ax):
    xmax = ax.dataLim.max[0]
    xrange = np.arange(1, xmax + 1, dtype='int16')
    ax.xaxis.set_ticks(xrange)

    ymin = ax.dataLim.min[1]
    ymax = ax.dataLim.max[1]
    yrange = ymax - ymin
    yticks = np.linspace(ymin - 0.05 * yrange, ymax + 0.05 * yrange, 15)
    ax.yaxis.set_ticks(yticks)
    return ax

class PlotLearningCurves(keras.callbacks.Callback):
    '''

    Class for plotting learning curves during keras training.
    Implemented as a callback to be transferred to model.fit method.

    Usage example:
    from aid_funcs.keraswrapper import PlotLearningCurves
    plot_curves_callback = PlotLearningCurves()
    model.fit(X, Y, nb_epoch=10, batch_size=10, callbacks=[plot_curves_callback])

    '''
    def __init__(self, metric_name='acc', model_name=''):
        self.metric_name=metric_name
        self.model_name = model_name
        self.metric = []
        self.val_metric = []
        self.loss = []
        self.val_loss = []
        self.file_name = 'l_curve_' + self.model_name + '_' + time.strftime("%H_%d_%m_%Y") + '.png'

        self.xdata = list([])  # holds the epochs for x axis

    def on_train_begin(self, logs={}):
        ''' initialize lists of loss and metrics and setup the figure'''

        # generating the general structure of the figure and plots
        # left-sided graph for the metric
        from plot import set_curr_fig_size
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        set_curr_fig_size(0.8)
        str_title = self.metric_name.replace("_", " ")
        str_title = str_title.title()
        ax1.set_title('Model ' + str_title)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Metric')
        ax1.set_axisbelow(True)
        ax1.yaxis.grid(color='gray', linestyle='dashed')
        # right sided graph for the loss
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_axisbelow(True)
        ax2.yaxis.grid(color='gray', linestyle='dashed')

        self.fig, self.ax1, self.ax2 = fig, ax1, ax2

    def on_epoch_end(self, epoch, logs={}):
        ''' Actual accumulation of data from logs and plotting '''
        self.metric.append(logs[self.metric_name])
        self.val_metric.append(logs['val_' + self.metric_name])
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        self.xdata.append(epoch+1)
        fig, ax1, ax2 = self.fig, self.ax1, self.ax2

        acc_line_train, = ax1.plot(self.xdata, self.metric, '-bo', label='Train')
        acc_line_val, = ax1.plot(self.xdata, self.val_metric, '-ro', label='Validation')
        ax1.legend(loc='upper left', handles=[acc_line_train, acc_line_val])
        ax1 = optimize_ticks(ax1)

        loss_line_train, = ax2.plot(self.xdata, self.loss, '-bo', label='Train')
        loss_line_val, = ax2.plot(self.xdata, self.val_loss, '-ro', label='Validation')
        ax2.legend(loc='upper left', handles=[loss_line_train, loss_line_val])
        ax2 = optimize_ticks(ax2)

        fig.canvas.draw()
        plt.pause(1e-6)
        fig.savefig(self.file_name)

