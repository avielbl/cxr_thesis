import numpy as np
import cv2

def square_image(img, padvalue = np.nan):
    r, c = img.shape
    if r == c:
        return img
    square_sz = max((r, c))
    out_img = np.ones((square_sz, square_sz), dtype=img.dtype) * padvalue
    out_img[:r, :c] = img
    return out_img



def im_rescale(img, new_min=0.0, new_max=1.0):
    """
    Changes intensity values of an image linearly to a new range

    :param img: image array
    :param new_min: new requested min value (default 0.0)
    :param new_max: new requested max value (default 1.0)
    :return: rescaled image in float32
    """

    img = img.astype('float32')
    curr_min = np.min(img)
    curr_max = np.max(img)

    unit_i = (img - curr_min) / (curr_max - curr_min)
    if new_min == 0.0 and new_max == 1.0:
        return unit_i

    out = (new_max - new_min) * unit_i + new_min
    return out


def imresize(img, newsz, interp=1):
    """
    Performing image resize to a given new size or by a given factor and by selecting the
    appropriate interpolation kernel

    :param img: the image to be resized (either grayscale or rgb)
    :param newsz: new shape in a tuple (new_rows, new_cols) or a scalar bigger than 0 as a resize factor
    :param interp: interpolation kernel 0=nearest neighbour, 1=bilinear (default), 2=bicubic
    :return: new resized image in a ndarray or -1 if failed
    """

    interpkernel = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_CUBIC
    }
    img = np.array(img)
    # Treat newsz as scaling factor
    if not isinstance(newsz, (list, tuple)):
        if newsz <= 0:
            raise ValueError('resizing factor should be greater than 0')
        return cv2.resize(img, (0,0), fx=newsz, fy=newsz, interpolation=interpkernel[interp])
    # Treat newsz as tuple shape
    else:
        revSize = [newsz[1], newsz[0]]
        return cv2.resize(img, tuple(revSize), interpolation=interpkernel[interp])


def resize_w_aspect(img, im_size, padvalue=0):
    """
    Performs image resize to square while maintaining aspect ratio of input and
    padding with zeros remaining portion of the squared output

    :param img: image array to be resize
    :param im_size: scalar value of output size
    :param padvalue: scalar value for constant padding of the output image (default to 0)
    :return: square image with width=length=im_size constant padded in the smaller dimension
    """

    sz = np.shape(img)
    ratio = sz[0] / sz[1]
    if ratio > 1:
        row_ratio = im_size / sz[0]
        column_size = int(sz[1] * row_ratio)
        new_sz = (im_size, column_size)
    else:
        column_ratio = im_size / sz[1]
        row_size = int(sz[0] * column_ratio)
        new_sz = (row_size, im_size)
    intype = img.dtype
    img = imresize(img, (new_sz[0], new_sz[1]))
    out = (np.ones((im_size, im_size)) * padvalue).astype(intype)
    out[0:new_sz[0], 0:new_sz[1]] = img
    return out


def safe_binary_morphology(mask, sesize=5, mode='close'):
    """
    Function to perform 'safe' binary morphology closing by padding with zeros input mask, performing the binary
    closing and then crop the padded area

    :param mask: binary 2d ndarray of the masked to be manipulated
    :param sesize: the size in pixels of the required structure element (disk shaped). default=3
    :param mode: name of required operation. either 'close' (default) or 'open'
    :return: binary 2d ndarray of the same shape as mask after manipulation
    """

    import skimage.morphology as morph

    pad_width = ((sesize, sesize), (sesize, sesize))
    padded_mask = np.pad(mask, pad_width, mode='constant')

    se = morph.disk(sesize)
    if mode == 'close':
        morphed_mask = morph.binary_closing(padded_mask, se)
    elif mode == 'open':
        morphed_mask = morph.binary_opening(padded_mask, se)
    elif mode == 'dilate':
        morphed_mask = morph.binary_dilation(padded_mask, se)
    elif mode == 'erode':
        morphed_mask = morph.binary_erosion(padded_mask, se)
    else:
        raise ValueError('Unknown mode: mode parameter accept either "close" or "open"')

    out = morphed_mask[sesize:-sesize, sesize:-sesize]
    return out
