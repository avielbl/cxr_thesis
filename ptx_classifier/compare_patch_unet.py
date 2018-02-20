import pickle
import matplotlib.pyplot as plt


with open('U-Net_WCE_roc.pkl', 'rb') as f:
    fpr_unet, tpr_unet, opt_ind_unet = pickle.load(f)
with open('patch_pixel_roc.pkl', 'rb') as f:
    fpr_patch, tpr_patch, opt_ind_patch = pickle.load(f)

fig = plt.figure(1)
plt.plot(fpr_unet, tpr_unet, label='FCN')
plt.plot(fpr_patch, tpr_patch, label='Patches')
plt.plot(fpr_unet[opt_ind_unet], tpr_unet[opt_ind_unet], 'ro')
plt.plot(fpr_patch[opt_ind_patch], tpr_patch[opt_ind_patch], 'ro')
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.legend(loc='upper right')
# plt.title('ROC curve {} (area = {:.1f}, opt thresh = {:.2f})'.format(title, 100 * roc_auc, opt_thresh))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.show()
plt.savefig('roc_analysis_compare_fcn-patch.png')
