import cv2
import glob
import math
import numpy as np
import matplotlib.pyplot as plt


cv_show = False
show_plot = True
view_points = False
img = []

def preprocess(image):
    image = cv2.imread(image) #convert image to numpy array of shape (h,w,c) -->here c is rgb values
    scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert rgb image to gray scale (h,w) (rgb 24 bits to 8 bits)
    #scaled = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return scaled

def blur(image):
    image = cv2.imread(image)
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(image, -1, kernel)
    cv2.imshow('gaussian_blur_image',dst)
    cv2.waitKey(0) #wait for user to press a key to dstroy a window
    cv2.destroyAllWindows()

def gaussian_blur_func(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0) #here (5,5) is size of the kernel, can be odd and positive
    return blur


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    #apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def show_images(images, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    x= len(images)/2
    cols = math.ceil(float(x))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Takes canny edge image as input and draws hough lines
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


if __name__ == '__main__':
    images = glob.glob('./data/*.png') # return list of all the images in folder data
    #for i in images:
    image = cv2.imread(images[2])
    img.append(image)
    gray_image = preprocess(images[0]) # to read only first image
    img.append(gray_image)
    gausian_blur = gaussian_blur_func(gray_image)
    img.append(gausian_blur)
    edge_canny = cv2.Canny(gausian_blur, 10, 200)
    img.append(edge_canny)
    #edge_canny = auto_canny(gausian_blur)
    imshape = image.shape
    xsize = imshape[1]
    ysize = imshape[0]
    vertices = np.array([[(0*xsize,ysize),(0 * xsize, 0.75* ysize), (0.5 * xsize, 0.5 * ysize), (0.65* xsize, 0.5 * ysize), (1*xsize, 0.75*ysize), (xsize, ysize)]], dtype=np.int32)  #

    if view_points:
        a = plt.imshow(image)
        plt.scatter(0*xsize,ysize)
        plt.scatter(0*xsize, 0.75*ysize)
        plt.scatter(0.5 * xsize, 0.5 * ysize)
        plt.scatter(0.65 * xsize, 0.5 * ysize)
        plt.scatter(1*xsize, 0.75*ysize)
        plt.scatter(xsize, ysize)
        plt.show()

    masked_image = region_of_interest(edge_canny,vertices)
    img.append(masked_image)
    line_image = hough_lines(masked_image, 1, np.pi/180, 20, 40, 25)
    img.append(line_image)
    lane_image = cv2.addWeighted(image, 0.8,line_image, 1., 0.)
    img.append(lane_image)

    #to see final image with lanes
    cv2.imshow('lane',lane_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if cv_show:
        for im in img:
            cv2.imshow('Images', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if show_plot:
        show_images(img)




