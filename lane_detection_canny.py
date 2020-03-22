import cv2
import glob
import math
import numpy as np
import matplotlib.pyplot as plt


cv_show = False
show_plot = False
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


def draw_lines(img, lines, color=[200, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    x_left = []
    y_left = []
    x_right = []
    y_right = []
    imshape = image.shape
    ysize = imshape[0]
    ytop = int(0.6 * ysize)  # need y coordinates of the top and bottom of left and right lane
    ybtm = int(ysize)  # to calculate x values once a line is found

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = float(((y2 - y1) / (x2 - x1)))
            if (slope > 0.5):  # if the line slope is greater than tan(26.52 deg), it is the left line
                x_left.append(x1)
                x_left.append(x2)
                y_left.append(y1)
                y_left.append(y2)
            if (slope < -0.5):  # if the line slope is less than tan(153.48 deg), it is the right line
                x_right.append(x1)
                x_right.append(x2)
                y_right.append(y1)
                y_right.append(y2)
    # only execute if there are points found that meet criteria, this eliminates borderline cases i.e. rogue frames
    if (x_left != []) & (x_right != []) & (y_left != []) & (y_right != []):
        left_line_coeffs = np.polyfit(x_left, y_left, 1)
        left_xtop = int((ytop - left_line_coeffs[1]) / left_line_coeffs[0])
        left_xbtm = int((ybtm - left_line_coeffs[1]) / left_line_coeffs[0])
        right_line_coeffs = np.polyfit(x_right, y_right, 1)
        right_xtop = int((ytop - right_line_coeffs[1]) / right_line_coeffs[0])
        right_xbtm = int((ybtm - right_line_coeffs[1]) / right_line_coeffs[0])
        cv2.line(img, (left_xtop, ytop), (left_xbtm, ybtm), color, thickness)
        cv2.line(img, (right_xtop, ytop), (right_xbtm, ybtm), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


if __name__ == '__main__':
    n = 0
    images = glob.glob('./data/*.png') # return list of all the images in folder data
    #to find lanes for all images
    for i in images:
        image = cv2.imread(i)
        #image = cv2.imread(images[0])
        #img.append(image)
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
        cv2.imwrite('_out/%02d.png' %n, lane_image)
        cv2.imshow('lane',lane_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        n+=1

        if cv_show:
            for im in img:
                cv2.imshow('Images', im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if show_plot:
            show_images(img)




