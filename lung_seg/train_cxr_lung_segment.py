from utilfuncs import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from aid_funcs.keraswrapper import get_unet, PlotLearningCurves, print_model_to_file
from aid_funcs.keraswrapper import dice_coef, dice_coef_loss
from keras.optimizers import Adam
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

##################################
# 1. Network training
##################################
# data initialization
im_size = params.im_size
print('Loading data...')
images_arr = load_images(params.training_data_dicom)
seg_map_arr = load_segmentation_maps(params.training_segmentation_maps)
save_pre_process_params(images_arr)
images_arr = pre_process_images(images_arr)

# splitting data to train and val sets manualy
val_idx = np.concatenate((np.arange(0, 18), np.arange(19, 26), np.arange(31, 41)), axis=0)
nb_val = np.shape(val_idx)[0]
nb_image = np.shape(images_arr)[0]
train_image_arr = []
train_seg_map_arr = []
val_images_arr = []
val_seg_map_arr = []
for i in range(nb_image):
    if i in val_idx:
        val_images_arr.append(images_arr[i])
        val_seg_map_arr.append(seg_map_arr[i])
    else:
        train_image_arr.append(images_arr[i])
        train_seg_map_arr.append(seg_map_arr[i])
train_image_arr = np.array(train_image_arr)
train_seg_map_arr = np.array(train_seg_map_arr)
val_images_arr = np.array(val_images_arr)
val_seg_map_arr = np.array(val_seg_map_arr)

print('Creating and compiling model...')
nb_epochs = 100
lr = 0.0001
optim_fun = Adam(lr=lr)
model = get_unet(params.im_size, lrelu_alpha=0.1,
                 filters=32, dropout_val=0.2,
                 loss_fun=dice_coef_loss, metrics=dice_coef, optim_fun=optim_fun)
print_model_to_file(model)

# Defining callbacks
model_file_name = 'lung_seg_model_' + time.strftime("%H_%M_%d_%m_%Y") + '.hdf5'
model_checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1, mode='min')
plot_curves_callback = PlotLearningCurves(metric_name='dice_coef')

print('-'*30)
print('Fitting model...')
print('-'*30)
model.fit(train_image_arr, train_seg_map_arr, batch_size=4, nb_epoch=100, verbose=1,
          validation_data=(val_images_arr, val_seg_map_arr),shuffle=True,
          callbacks=[plot_curves_callback, model_checkpoint, early_stopping, reduce_lr_on_plateau])
print("Done!")

###############################################################
# 2. Finding optimal threshold on scores based on ROC analysis
###############################################################
scores_arr = np.ndarray((nb_val, im_size, im_size))

# generating predictions ofr val set
for i in range(nb_val):
    start_time = time.time()
    img = val_images_arr[i]
    img = np.reshape(img, (1, 1, im_size, im_size))
    scores = model.predict(img, verbose=0)
    scores = np.squeeze(scores)
    scores_arr[i] = scores

# calculating ROC per pixel
fpr, tpr, thresh = roc_curve(val_seg_map_arr.flatten(), scores_arr.flatten())
roc_auc = auc(fpr, tpr)
dist_to_opt = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
opt_ind = np.argmin(dist_to_opt)
opt_thresh = thresh[opt_ind]

# saving new threshold
set_optimal_thresh(opt_thresh)

# plotting the roc
plt.figure()
plt.plot(fpr, tpr, label='ROC')
plt.plot(fpr, thresh, label='Threshold')
plt.plot(fpr[opt_ind], tpr[opt_ind], 'ro', label='Optimal thresh')
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.legend(loc='upper right')
plt.title('ROC curve (area = %0.2f, opt thresh = %0.2f)' % (100 * roc_auc, opt_thresh))

plt.savefig('roc analysis.png')