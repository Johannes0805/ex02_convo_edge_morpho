import cv2
from PIL import Image
import numpy as np
from morphological import show_image

def make_kernel(ksize, sigma):
    radius = ksize // 2
    # create an unweighted kernel filled with zeros
    kernel = np.zeros((ksize, ksize))

    for i in range(ksize):
        for j in range(ksize):
            x = j - radius
            y = i - radius
            # used the fomula given in the lecture to create the kernel with right values
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # normalize kernel, so that it sums up to 1
    kernel /= np.sum(kernel)
    return kernel

def slow_convolve(arr, k):

    k = np.flip(k, (0, 1))

    out_img = np.zeros_like(arr, dtype=float)
    kh, kw = k.shape
    rh = kh // 2
    rw = kw // 2

    # checking whether the image is grayscale or rgb
    if len(arr.shape) == 2:
        height, width = arr.shape
        # padding the image array with zeros so that the kernel also evaluates the edges of the img
        padded = np.pad(arr,((rh,rh), (rw,rw)), mode='constant', constant_values=0)
        for i in range(height):
            for j in range(width):
                # creating a kernel sized copy
                patch = padded[i: (i + kh), j: (j + kw)]
                # calculating the weighted patch
                w_patch = patch * k
                # summing up all values of the weighted area into the pixel (i,j)
                out_img[i, j] = np.sum(w_patch)
    else:
        # all same like grayscale but done three times due to the three color channels
        height, width, channel = arr.shape
        padded = np.pad(arr,((rh,rh), (rw,rw), (0,0)), mode='constant', constant_values=0)
        for i in range(height):
            for j in range(width):
                for c in range(channel):
                    patch = padded[i: (i + kh), j: (j + kw), c]
                    w_patch = patch * k
                    out_img[i, j, c] = np.sum(w_patch)

    return out_img


if __name__ == '__main__':
    k = make_kernel(25, 1.6)  # todo: find better parameters original: ksize 3, sigma 1
    
    # TODO: chose the image you prefer
    # im = np.array(Image.open('data/input1.jpg'))
    # im = np.array(Image.open('data/input2.jpg'))

    im = np.array(Image.open('data/input3.jpg'))

    ### own tests ###
    o_img = slow_convolve(im, k).astype(np.uint8)
    o_img_save = cv2.cvtColor(o_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img=o_img_save, filename="data/input3_convoluted.png")
    show_image(o_img, "input3_convolved")


    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warm-up exercise?), convert
    #       the array to np.unit8, and save the result

    im = np.array(Image.open('data/input3.jpg'))

    blurred = slow_convolve(im, k)
    details = im - blurred
    sharpened = im + details
    sharpened = np.clip(sharpened, 0, 255)
    sharpened = sharpened.astype(np.uint8)
    show_image(sharpened, "input3_sharpened")