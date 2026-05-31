import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
#import cv2

#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """

    # Code reused from convo.py make_kernel #
    radius = ksize // 2 # calculate the middle of the kernel

    kernel = np.zeros((ksize, ksize))

    for i in range(ksize):
        for j in range(ksize):
            x = j - radius
            y = i - radius
            # used the fomula given in the lecture to create the kernel with right values
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    # normalize kernel, so that it sums up to 1
    kernel /= np.sum(kernel)
    # convolve the image
    filtered = convolve(img_in, kernel)

    return kernel, filtered.astype(int) # converted back to int as the test failed if not done


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # create the kernel like described in the lecture
    gx_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # flipped this kernel as convolve does flip it as well
    gy_kernel = np.flip(np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]))

    # convolve the image with both kernels
    gx = convolve(img_in, gx_kernel).astype(int)
    gy = convolve(img_in, gy_kernel).astype(int)

    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # applied fomulas given in the lecture slides
    return np.sqrt(gx**2 + gy**2).astype(int), np.arctan2(gy, gx)

def convertAngle(angle):
    """ compute nearest matching angle 
    :param angle: in radians 
    :return: nearest match of {0, 45, 90, 135} 
    """
    m_angle = 45 / 2
    # convert radiant to degrees
    angle = np.degrees(angle)
    # handling angles larger than 180
    angle = angle % 180

    if 0 <= angle < m_angle or angle >= (135 + m_angle):
        angle_snaped = 0
    elif m_angle <= angle < (90 - m_angle):
        angle_snaped = 45
    elif (90 - m_angle) <= angle < (90 + m_angle):
        angle_snaped = 90
    else:
        angle_snaped = 135
    return angle_snaped


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g: (np.ndarray) contains the strength of the gradient
    :param theta: 2d image (np.ndarray) contains the direction of the gradient
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above

    x_len, y_len = g.shape
    res = np.zeros_like(g) # create a new array sized like the g matrix
    # g includes padding that should not be considered in the result
    for x in range(1, x_len - 1):
        for y in range(1, y_len - 1):
            conv_angle = convertAngle(theta[x,y])
            # case: horizontal, check left and right
            if conv_angle == 0:
                nb1 = g[x, y-1]
                nb2 = g[x, y+1]
            # case: left low and right above
            elif conv_angle == 45:
                nb1 = g[x-1, y+1]
                nb2 = g[x+1, y-1]
            # case: vertical, check above and below
            elif conv_angle == 90:
                nb1 = g[x-1, y]
                nb2 = g[x+1, y]
            # case: left up and right below
            else:
                nb1 = g[x-1, y-1]
                nb2 = g[x+1, y+1]
            # value can be kept if it is larger than both of his neighbors
            if g[x, y] >= nb1 and g[x, y] >= nb2:
                res[x, y] = g[x, y]
            else:
                res[x, y] = 0
    return res


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lecture's hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """

    x_len, y_len = max_sup.shape
    tresh = np.zeros_like(max_sup)
    res = np.zeros_like(max_sup)

    # Classifying all pixels as weak(0), normal(1) or strong(2) and saving them in a "map" tresh
    for x in range(x_len):
        for y in range(y_len):
            p_value = max_sup[x,y]
            if p_value <= t_low:
                tresh[x,y] = 0
            elif p_value <= t_high:
                tresh[x,y] = 1
            else:
                tresh[x,y] = 2

    # Setting the correct hypothesis values using thresh
    for x in range(x_len):
        for y in range(y_len):

            if tresh[x, y] == 2:
                res[x,y] = 255
                # search for neighbors with minimum normal values
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        # skip weak neighbors
                        if dx == 0 and dy == 0:
                            continue
                        # check that the neighbors are within the img and not outside
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < x_len and 0 <= ny < y_len:
                            if tresh[nx, ny] >= 1:
                                res[nx, ny] = 255
    return res


def canny(img):
    # apply gaussian filter to img
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result

"""
if __name__ == '__main__':
    im = cv2.imread("data/input3.jpg")
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    canny(np.array(im))
"""