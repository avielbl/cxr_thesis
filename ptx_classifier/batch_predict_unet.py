from aid_funcs.keraswrapper import load_model
from misc import load_from_h5, roc_plotter
from prep_data_for_unet import prep_set
from utils import *



def analyze_performance(model=None, val_data = None, model_name=''):
    if model is None:
        # model = load_model('ptx_model_unet.hdf5')
        model = load_model('ptx_model_unet.hdf5', custom_objects='dice_coef_loss')
    if val_data is None:
        _, val_data_lst = process_and_augment_data()
        val_imgs_arr, val_masks_arr = prep_set(val_data_lst)
    else:
        val_imgs_arr = val_data[0]
        val_masks_arr = val_data[1]

    lung_masks_arr = load_from_h5(os.path.join(training_path, 'db_lung_masks_arr.h5'))

    scores = model.predict(val_imgs_arr, batch_size=5, verbose=1)

    # calculating ROC per pixel
    lung_flatten = np.array(lung_masks_arr).flatten()
    lung_idx = np.where(lung_flatten > 0)
    scores_vec = scores.flatten()
    scores_vec = scores_vec[lung_idx]
    gt_vec = np.uint8(val_masks_arr.flatten())
    gt_vec = gt_vec[lung_idx]
    roc_plotter(gt_vec, scores_vec, model_name)

    pred_maps = np.zeros_like(scores, dtype=np.uint8)
    pred_maps[scores > 0.11] = 255

    nb_val = val_imgs_arr.shape[0] + 1
    plt.figure()
    if os.path.isdir('results') is False:
        os.mkdir('results')
    for i in range(nb_val):
        img = np.squeeze(val_imgs_arr[i])
        pred = np.squeeze(pred_maps[i]) * np.squeeze(lung_masks_arr[i])

        gt = np.squeeze(val_masks_arr[i])

        show_image_with_overlay(img, overlay=gt, overlay2=pred)
        plt.axis('off')
        out_path = 'results\\' + str(i) + '.png'
        plt.savefig(out_path, bbox_inches='tight')
        print('saved image {}/{}'.format(i, nb_val))
