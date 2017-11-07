from aid_funcs.misc import roc_plotter
from aid_funcs.keraswrapper import load_model
from misc import load_from_h5
from utils import *

val_data_labels = load_from_h5(os.path.join(training_path, 'val_labels.h5'))
val_data_patches = load_from_h5(os.path.join(training_path, 'val_patches.h5'))

model = load_model(model_path)

nb_val = val_data_labels.shape[0]
scores = model.predict(val_data_patches, batch_size=1000, verbose=1)[:,1]

roc_plotter(val_data_labels, scores, 'Patch_based')
#
# # calculating ROC per pixel
# fpr, tpr, thresh = roc_curve(val_data_labels, scores)
# roc_auc = auc(fpr, tpr)
# dist_to_opt = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
# opt_ind = np.argmin(dist_to_opt)
# opt_thresh = thresh[opt_ind]
#
# # plotting the roc
# plt.figure(1)
# plt.plot(fpr, tpr, label='ROC')
# # plt.plot(fpr, thresh, label='Threshold')
# plt.plot(fpr[opt_ind], tpr[opt_ind], 'ro', label='Optimal thresh')
# plt.minorticks_on()
# plt.grid(b=True, which='both')
# plt.legend(loc='upper right')
# plt.title('ROC curve (area = %0.2f, opt thresh = %0.2f)' % (100 * roc_auc, opt_thresh))
# plt.savefig('roc analysis.png')