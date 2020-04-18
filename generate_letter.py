import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
import scipy
from scipy import ndimage
from scipy.interpolate import griddata

import cv2

import preprocess

if __name__ == "__main__":
	# DEFINE AND LOAD FONT
	script_root = '/Users/junkyungkim/Documents/PycharmProjects/cluttered_nist'
	fontnames = ['FUTRFW.ttf',
				 'Instruction.otf',
				 'absender1.ttf',
				 '5Identification-Mono.ttf',
				 '7Segment.ttf',
				 'VCR_OSD_MONO_1.001.ttf',
				 'Instruction.otf',
				 'Segment16B Regular.ttf']
	std_fontsizes = [225, 240, 225, 150, 255, 255, 255, 255]
	std_thin_iters = [6, 15, 4, 9, 9, 2]
	scale = 1  # 0.5

	for fontname, std_fontsize, std_thin_iter in zip(fontnames, std_fontsizes, std_thin_iters):
		std_fontsize = int(std_fontsize*scale)
		std_thin_iter = int(std_thin_iter*scale)

		font = ImageFont.truetype(os.path.join(script_root,'fonts',fontname), std_fontsize)

		# RENDER
		img=Image.new("RGBA", (2500, 300), (255, 255, 255))
		draw = ImageDraw.Draw(img)
		draw.text((0, 0), "ABCDEFGXYZ", (0, 0, 0), font=font)
		draw = ImageDraw.Draw(img)

		# MORPHOLOGICAL POSTPROC (FOR CONSTNAT STROKE THICKNESS)
		img = 255 - np.mean(np.array(img), axis=2)
		binary = img > 128
		# img_closed = scipy.ndimage.binary_closing(binary.astype(np.int), iterations=20)##np.maximum(iterations / 2, 1))
		img_eroded = (scipy.ndimage.morphology.binary_erosion(binary, iterations=std_thin_iter) * 255).astype(np.uint8)

		landscape = preprocess.generate_distortion_mask(img_eroded, sigma=[4000,2000], num_centers=[30,20])
		warped = preprocess.custom_warp(img_eroded, landscape, power=0.07)
		# img_dist = img_eroded
		# distCoeffs = [-.1, 1.0, 1.0, 1.0]
		# focal_length = [1000, 1000]
		# for coord in [[400,100],[500,150],[600,200]]:
		# 	distCoeffs[0] = distCoeffs[0]*-1
		# 	img_dist = custom_fisheye(img_dist, coord, distCoeffs, focal_length)

		# import preprocess
		# im_pixelated = preprocess.pixelate_obj(img_eroded, [10 * scale, 10 * scale], 0.1, 5 * scale, ignore_fit=True)
		plt.subplot(211);plt.imshow(binary, cmap='gray')
		plt.subplot(212);plt.imshow(warped, cmap='gray')
		plt.show()

		# thinned = zhangSuen(binary)

		# plt.subplot(121)
		# plt.imshow(img)
		# plt.subplot(122)
		# plt.imshow(thinned)
		# plt.show()