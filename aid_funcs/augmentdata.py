
def elastic_transform(image, alpha, sigma, mask=None, random_state=None):
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.ndimage.filters import gaussian_filter

    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(random_state)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    if mask is None:
        return map_coordinates(image, indices, order=1).reshape(shape)
    else:
        return (
            map_coordinates(image, indices, order=1).reshape(shape),
            map_coordinates(mask, indices, order=1).reshape(shape))


def rand_gamma_adjust(img, gamma_range=(0.6, 1.4)):
    from skimage import exposure
    from random import uniform
    gamma = uniform(*gamma_range)
    curr_min, curr_max = np.nanmin(img), np.nanmax(img)
    img = im_rescale(img)
    gamma_adjusted = exposure.adjust_gamma(img, gamma)
    gamma_adjusted = im_rescale(gamma_adjusted, curr_min, curr_max)
    return gamma_adjusted


def rand_im_rotate(img, mask=None, angle_range=(-7, 7)):
    from random import uniform
    angle = uniform(*angle_range)
    M = cv2.getRotationMatrix2D((im_size/2, im_size/2), angle, 1)
    img_out = cv2.warpAffine(img, M, (im_size, im_size))
    if mask == None:
        return img_out
    else:
        mask_out = cv2.warpAffine(mask, M, (im_size, im_size))
        return (img_out, mask_out)


def create_2d_gaussian(dim, sigma):
    """
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma

    :param dim: integer denoting a side (1-d) of gaussian kernel
    :type dim: int
    :param sigma: the standard deviation of the gaussian kernel
    :type sigma: float

    :returns: a numpy 2d array
    """

    # check if the dimension is odd
    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # initialize the kernel
    kernel = np.zeros((dim, dim), dtype=np.float16)

    # calculate the center point
    center = dim / 2

    # calculate the variance
    variance = sigma ** 2

    # calculate the normalization coefficeint
    coeff = 1. / (2 * variance)

    # create the kernel
    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val ** 2 + y_val ** 2
            denom = 2 * variance

            kernel[x, y] = coeff * np.exp(-1. * numerator / denom)

    # normalise it
    return kernel / sum(sum(kernel))


def elastic_transform(image, kernel_dim=65, sigma=64, alpha=30, negated=False, gridsize=5):
    """
    This method performs elastic transformations on an image by convolving
    with a gaussian kernel.
    NOTE: Image dimensions should be a sqaure image

    :param image: the input image
    :type image: a numpy nd array
    :param kernel_dim: dimension(1-D) of the gaussian kernel
    :type kernel_dim: int
    :param sigma: standard deviation of the kernel
    :type sigma: float
    :param alpha: a multiplicative factor for image after convolution
    :type alpha: float
    :param negated: a flag indicating whether the image is negated or not
    :type negated: boolean
    :returns: a nd array transformed image
    """

    # convert the image to single channel if it is multi channel one
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check if the image is a negated one
    if not negated:
        image = 255 - image

    # check if the image is a square one
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image should be of sqaure form")

    # check if kernel dimesnion is odd
    if kernel_dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # create an empty image
    result = np.zeros(image.shape)

    # create random displacement fields
    displacement_field_x = np.array([[(2*ranf()-1) for x in range(image.shape[0])]
                                     for y in range(image.shape[1])]) * alpha
    displacement_field_y = np.array([[(2*ranf()-1) for x in range(image.shape[0])]
                                     for y in range(image.shape[1])]) * alpha
    # displacement_field_x = np.zeros((gridsize+2, gridsize+2))
    # displacement_field_y = np.zeros((gridsize+2, gridsize+2))

    # displacement_field_x = np.array([[randint(-1, 2) for x in range(5)]
    #                                  for y in range(5)]) * alpha
    # displacement_field_y = np.array([[randint(-1, 2) for x in range(5)]
    #                                  for y in range(5)]) * alpha

    # displacement_field_x[1:-1, 1:-1] = (rand(gridsize, gridsize) * 2 - 1) * alpha
    # displacement_field_y[1:-1, 1:-1] = (rand(gridsize, gridsize) * 2 - 1) * alpha
    #
    # displacement_field_x = cv2.resize(displacement_field_x, (image.shape[0], image.shape[1]))
    # displacement_field_y = cv2.resize(displacement_field_y, (image.shape[0], image.shape[1]))
    # create the gaussian kernel
    kernel = create_2d_gaussian(kernel_dim, sigma)

    # convolve the fields with the gaussian kernel
    displacement_field_x = convolve2d(displacement_field_x, kernel, boundary='symm')
    displacement_field_y = convolve2d(displacement_field_y, kernel, boundary='symm')

    # make the distortrd image by averaging each pixel value to the neighbouring
    # four pixels based on displacement fields

    for row in range(image.shape[1]):
        for col in range(image.shape[0]):
            low_ii = row + int(math.floor(displacement_field_x[row, col]))
            high_ii = row + int(math.ceil(displacement_field_x[row, col]))

            low_jj = col + int(math.floor(displacement_field_y[row, col]))
            high_jj = col + int(math.ceil(displacement_field_y[row, col]))

            if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] - 1 \
                    or high_jj >= image.shape[0] - 1:
                continue

            res = image[low_ii, low_jj] / 4 + image[low_ii, high_jj] / 4 + \
                  image[high_ii, low_jj] / 4 + image[high_ii, high_jj] / 4

            result[row, col] = res

    # if the input image was not negated, make the output image also a non
    # negated one
    if not negated:
        result = 255 - result

    return result
