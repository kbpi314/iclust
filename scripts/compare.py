#!/usr/bin/env python
from __future__ import division

import click

# From https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
# Other interesting reads:
#   https://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images
#   https://stackoverflow.com/questions/5101004/python-code-for-earth-movers-distance
#   https://en.wikipedia.org/wiki/Point_set_registration
#   https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/


from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os.path

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='0.1')

@click.option('-p', '--path', required=False,
              type=click.Path(exists=True),
              help='Input  default config file path')

def compare(path):
    """ 
    """

    ## When comparing images, think if:
    ##  1) use SSIM or MSE
    ##  2) use natural or log scale (to 'separate' points close to each other)
    ##  3) use open or closed circles to represent points

    files = glob.glob(path + '*.jpeg')
    all_img = []
    all_names = []
    
    for f in files:
        t = cv2.imread(f)
        all_names.append(os.path.basename(f))
        all_img.append(t)

    print all_names
        
    # load the images -- the original, the original + contrast,
    # and the original + photoshop
    #original = cv2.imread("images/jp_gates_original.png")
    #contrast = cv2.imread("images/jp_gates_contrast.png")
    #shopped = cv2.imread("images/jp_gates_photoshopped.png")
 
    # convert the images to grayscale
    #original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    #shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

    # initialize the figure
    #fig = plt.figure("Images")
    #images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
 
    # loop over the images
    #for (i, (name, image)) in enumerate(images):
	# show the image
	#ax = fig.add_subplot(1, 3, i + 1)
	#ax.set_title(name)
	#plt.imshow(image, cmap = plt.cm.gray)
	#plt.axis("off")
 
    # show the figure
    #plt.show()
 
    # compare the images
    mse_vals = []
    ssim_vals = []
    corr_vals = []
    for (i,img) in enumerate(all_img):
        mv = []
        sv = []
        cv = []
        for j in range(len(all_img)):
            m, s = compare_images(all_img[i], all_img[j], str(i)+' '+str(j))
            mv.append(str(m))
            sv.append(str(s))
            corr_diff = np.absolute(float(all_names[i].split('_')[-1].split('.jpeg')[0])) - np.absolute(float(all_names[j].split('_')[-1].split('.jpeg')[0]))
            cv.append(str(corr_diff))
        mse_vals.append(mv)
        ssim_vals.append(sv)
        corr_vals.append(cv)

    fo = open(path + 'mse.txt','w')
    fo.write('\t'.join(all_names)+'\n')
    for v in mse_vals:
        fo.write('\t'.join(v)+'\n')
    fo.close()
    
    fo = open(path + 'ssim.txt','w')
    fo.write('\t'.join(all_names)+'\n')    
    for v in ssim_vals:
        fo.write('\t'.join(v)+'\n')
    fo.close()

    fo = open(path + 'corr.txt','w')
    fo.write('\t'.join(all_names)+'\n')    
    for v in corr_vals:
        fo.write('\t'.join(v)+'\n')
    fo.close()
    #compare_images(t1, t2, 'FP_1008_1173_0618 vs FP_1015_101_0995.jpeg')
    #compare_images(t1, t3, 'FP_1008_1173_0618 vs TP_100_885_0775.jpeg')
    #compare_images(t2, t3, 'FP_1015_101_0995.jpeg vs TP_100_885_0775.jpeg')
    
    #compare_images(original, original, "Original vs. Original")
    #compare_images(original, contrast, "Original vs. Contrast")
    #compare_images(original, shopped, "Original vs. Photoshopped")
    
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, multichannel=True)
    return m, s
    # print m, s, title
    # setup the figure
    #fig = plt.figure(title)
    #plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    #ax = fig.add_subplot(1, 2, 1)
    #plt.imshow(imageA, cmap = plt.cm.gray)
    #plt.axis("off")
    # show the second image
    #ax = fig.add_subplot(1, 2, 2)
    #plt.imshow(imageB, cmap = plt.cm.gray)
    #plt.axis("off")
    # show the images
    #plt.show()


if __name__ == "__main__":
    compare()
