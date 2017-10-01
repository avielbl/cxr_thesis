import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import keras
from sklearn.metrics import roc_curve, auc

from aid_funcs.keraswrapper import load_model
from utils import *

val_data_patches = pickle.load(open(os.path.join(training_path, 'val_patches.pkl'), 'rb'))
val_data_labels = pickle.load(open(os.path.join(training_path, 'val_labels.pkl'), 'rb'))

model = load_model(model_path)

nb_val = val_data_labels.shape[0]
# scores = model.predict(val_data_patches, batch_size=1000, verbose=1)
scores = np.zeros((nb_val,), dtype=np.uint8)
for i, patch in enumerate(val_data_patches):
    scores[i] = model.predict(np.expand_dims(patch, 0))
    if i % 100 == 0:
        print('Predicted {}/{}'.format(i, nb_val))

# calculating ROC per pixel
fpr, tpr, thresh = roc_curve(val_data_labels, scores)
roc_auc = auc(fpr, tpr)
dist_to_opt = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
opt_ind = np.argmin(dist_to_opt)
opt_thresh = thresh[opt_ind]

# plotting the roc
plt.figure(1)
plt.plot(fpr, tpr, label='ROC')
plt.plot(fpr, thresh, label='Threshold')
plt.plot(fpr[opt_ind], tpr[opt_ind], 'ro', label='Optimal thresh')
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.legend(loc='upper right')
plt.title('ROC curve (area = %0.2f, opt thresh = %0.2f)' % (100 * roc_auc, opt_thresh))
plt.savefig('roc analysis.png')