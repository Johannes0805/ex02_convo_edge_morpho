import math
import cv2
from PIL import Image
import numpy as np

from morphological import show_image, save_binary

"""
def make_kernel(ksize, sigma):
   # implement the Gaussian kernel here
   r = ksize // 2

   kernel = np.zeros((ksize, ksize))
   for i in range(ksize):
       for j in range(ksize):
           x = i - ksize
           y = j - ksize
           kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

   kernel = kernel / np.sum(kernel)

   ###############################################################

   y, x = np.mgrid[-r : r + 1, -r : r + 1]

   kernel = np.exp(x**2 + y**2 / (2 * sigma**2))

   kernel = kernel / np.sum(kernel)



   return kerneldef make_kernel(ksize, sigma):
   # implement the Gaussian kernel here
   r = ksize // 2
   kernel = np.zeros((ksize, ksize))
   for i in range(ksize):
       for j in range(ksize):
           x = i - ksize
           y = j - ksize
           kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

   kernel = kernel / np.sum(kernel)
   ###############################################################

   y, x = np.mgrid[-r : r + 1, -r : r + 1]

   kernel = np.exp(x**2 + y**2 / (2 * sigma**2))

   kernel = kernel / np.sum(kernel)


   return kernel
   """


def make_kernel(ksize, sigma):
    radius = ksize // 2

    kernel = np.zeros((ksize, ksize))

    for i in range(ksize):
        for j in range(ksize):
            x = j - radius
            y = i - radius
            # used the fomula given in the lecture to create the kernel with right values
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    kernel /= np.sum(kernel)  # normalize kernel, so that it sums up to 1
    return kernel






def slow_convolve(arr, k):
    k = np.flip(k, (0, 1))

    out_img = np.zeros_like(arr, dtype=float)
    kh, kw = k.shape
    rh = kh // 2
    rw = kw // 2

    if len(arr.shape) == 2:
        height, width = arr.shape
        # print("kernel shape:", k.shape)
        padded = np.pad(arr,((rh,rh), (rw,rw)), mode='constant', constant_values=0)
        for i in range(height):
            for j in range(width):
                patch = padded[i:i + kh, j:j + kw]
                out_img[i, j] = np.sum(patch * k)
    else:
        height, width, channel = arr.shape
        padded = np.pad(arr,((rh,rh), (rw,rw), (0,0)), mode='constant', constant_values=0)
        for i in range(height):
            for j in range(width):
                for c in range(channel):
                    patch = padded[i:i + kh, j:j + kw, c]
                    out_img[i, j, c] = np.sum(patch * k)

    """
    for i in range(width):
        for j in range(height):
            for c in range(channel):
                value = 0
                for u in range(ksize):
                    for v in range(ksize):
                            value += k[u, v] * padded[i + u, j + v, c]
                out_img[i, j, c] = value
    """

    out_img = np.clip(out_img, 0, 255)

    return out_img


if __name__ == '__main__':
    k = make_kernel(25, 1.6)  # todo: find better parameters original: ksize 3, sigma 1
    
    # TODO: chose the image you prefer
    # im = np.array(Image.open('data/input1.jpg'))
    # im = np.array(Image.open('data/input2.jpg'))

    im = np.array(Image.open('data/input3.jpg'))
    o_img = slow_convolve(im, k).astype(np.uint8)
    o_img_save = cv2.cvtColor(o_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img=o_img_save, filename="data/input3_convoluted.png")
    show_image(o_img, "input3_convolved")


    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
