import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from scipy import ndimage
import skimage
from skimage.morphology import medial_axis

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
import scipy
import scipy.stats
import cv2

def pixelate_obj(img, patch_size, threshold, p_omission, sig_shift, jitter_type='gaussian', ignore_patch_fit=True):
    if not ignore_patch_fit:
        if (img.shape[0] % patch_size[0] != 0) | (img.shape[1] % patch_size[1] != 0):
            return ValueError('pixelate: img size must be divisible by patch_size')
    canvas = np.zeros_like(img)
    y_list = np.arange(0, img.shape[0] - patch_size[0] + 1, patch_size[0])
    x_list = np.arange(0, img.shape[1] - patch_size[1] + 1, patch_size[1])
    max_val = np.max(img)
    for iy in y_list:
        for ix in x_list:
            if np.random.uniform() < p_omission:
                continue  # apply omission
            else:
                ## can work either on fused img (2d) or segmented master img (3d)
                if (len(img.shape) == 3) & (img.shape[2] > 1): # segmented
                    patch = img[iy:iy + patch_size[0], ix:ix + patch_size[1], :]
                    val = (np.mean(patch/max_val, axis=(0, 1)) > threshold).astype(np.int)
                    draw = np.any(val==1)
                else: # fused
                    patch = img[iy:iy + patch_size[0], ix:ix + patch_size[1]]
                    val = (np.mean(patch/max_val) > threshold).astype(np.int)  # pixelate patch
                    draw = (val==1)
                if draw:  # RENDER PATCH
                    if jitter_type == 'uniform':
                        np.random.uniform(low=0.0, high=1.0, size=None)
                        iy_delta = np.random.uniform(-2 * sig_shift, 2 * sig_shift)
                        ix_delta = np.random.uniform(-2 * sig_shift, 2 * sig_shift)
                    elif jitter_type == 'gaussian':
                        iy_delta = np.maximum(np.minimum(np.random.normal(0, sig_shift), sig_shift * 2),
                                              -sig_shift * 2).astype(np.int)
                        ix_delta = np.maximum(np.minimum(np.random.normal(0, sig_shift), sig_shift * 2),
                                              -sig_shift * 2).astype(np.int)
                    else:
                        return ValueError('pixelate: jitter_type should be uniform or gaussian')
                    iy_perturbed = np.maximum(np.minimum(iy + iy_delta, img.shape[0] - patch_size[0]), 0)
                    ix_perturbed = np.maximum(np.minimum(ix + ix_delta, img.shape[1] - patch_size[1]), 0)
                    if (len(img.shape) == 3) & (img.shape[2] > 1): # segmented
                        for i_obj in range(img.shape[2]):
                            if val[i_obj] >0:
                                canvas[iy_perturbed:iy_perturbed + patch_size[0], ix_perturbed:ix_perturbed + patch_size[1],i_obj] = val[i_obj]
                    else: #fused
                        canvas[iy_perturbed:iy_perturbed + patch_size[0], ix_perturbed:ix_perturbed + patch_size[1]] = val
                else:
                    continue
    return canvas


def add_pixelated_noise(img, patch_size, p_omission, sig_shift, ignore_fit=True):
    if not ignore_fit:
        if (img.shape[0] % patch_size[0] != 0) | (img.shape[1] % patch_size[1] != 0):
            return ValueError('pixelate: img size must be divisible by patch_size')
    canvas = np.zeros_like(img)
    y_list = np.arange(0, img.shape[0] - patch_size[0] + 1, patch_size[0])
    x_list = np.arange(0, img.shape[1] - patch_size[1] + 1, patch_size[1])
    for iy in y_list:
        for ix in x_list:
            if np.random.uniform() < p_omission:
                continue  # apply omission
            else:
                patch = img[iy:iy + patch_size[0], ix:ix + patch_size[1]]
                val = (np.mean(patch) > 0.5).astype(np.int)  # pixelate patch
                if val == 1:  # RENDER PATCH
                    iy_delta = np.maximum(np.minimum(np.random.normal(0, sig_shift), sig_shift * 2),
                                          -sig_shift * 2).astype(np.int)
                    ix_delta = np.maximum(np.minimum(np.random.normal(0, sig_shift), sig_shift * 2),
                                          -sig_shift * 2).astype(np.int)
                    iy_perturbed = np.maximum(np.minimum(iy + iy_delta, img.shape[0] - patch_size[0]), 0)
                    ix_perturbed = np.maximum(np.minimum(ix + ix_delta, img.shape[1] - patch_size[1]), 0)
                    canvas[iy_perturbed:iy_perturbed + patch_size[0], ix_perturbed:ix_perturbed + patch_size[1]] = 1
                else:
                    continue
    return canvas


def rotate_n_scale(im, rotation=0, scale=[1.0, 1.0], pad_val=0):  # ROTATION SPECIFIED IN DEGS
    # PAD
    tilt = np.mod(rotation, 90)
    expansion = np.sqrt(2) * np.sin((45 + tilt) * np.pi / 180)
    hpad = int(np.maximum(expansion + 0.05 - 1, 0.) * im.shape[0] / 2)
    wpad = int(np.maximum(expansion + 0.05 - 1, 0.) * im.shape[1] / 2)
    im_padded = np.pad(im, pad_width=((hpad, hpad), (wpad, wpad)), mode='constant',
                       constant_values=((pad_val, pad_val), (pad_val, pad_val)))

    current_imsize = [int(im_padded.shape[0]), int(im_padded.shape[1])]
    center = [current_imsize[0] / 2, current_imsize[1] / 2]

    # TRANSFORM (rotation in degrees)
    S_mat = np.float32([[scale[1], 0, -(scale[1]-1)*center[1]], [0, scale[0], -(scale[0]-1)*center[0]]])
    im_out = cv2.warpAffine(im_padded, S_mat, (current_imsize[1], current_imsize[0]))
    if rotation != 0:
        R_mat = cv2.getRotationMatrix2D((current_imsize[1] / 2, current_imsize[0] / 2), rotation, 1.0)
        im_out = cv2.warpAffine(im_out, R_mat, (current_imsize[1], current_imsize[0]))
        # plt.subplot(121);plt.imshow(im);plt.subplot(122);plt.imshow(im_out);plt.show()
    return im_out


def shear(im, hshear=0., wshear=0., pad_val=0):  # ROTATION SPECIFIED IN DEGS
    # PAD
    hpad = np.abs(int(im.shape[1] * hshear / 2))
    wpad = np.abs(int(im.shape[0] * wshear / 2))

    im_padded = np.pad(im, pad_width=((hpad, hpad), (wpad, wpad)), mode='constant',
                       constant_values=((pad_val, pad_val), (pad_val, pad_val)))
    # TRANSFORM
    current_imsize = im_padded.shape
    M = np.array([[1, wshear, -int(im_padded.shape[0] * wshear / 2)],
                  [hshear, 1, -int(im_padded.shape[1] * hshear / 2)]]).astype(np.float)
    im_out = cv2.warpAffine(im_padded, M, (current_imsize[1], current_imsize[0]))
    return im_out


def measure_obj_thickness(im):
    cell_coords = np.nonzero(im)
    # cell_coords = np.transpose(np.array([cell_coords[0], cell_coords[1]])).astype(np.float)
    # cell_coords -= np.mean(cell_coords.astype(np.float), axis=0)
    # covariance_mat = np.cov(cell_coords, rowvar=False)
    # evals, evecs = LA.eigh(covariance_mat)
    # evals = evals[np.argsort(evals)[::-1]]  # descending order
    # area = cell_coords.shape[0]
    # spread = np.sqrt(evals[0]*evals[1]
    if np.max(np.max(im)) == 0 :
        raise ValueError('image is all 0')
    else:
        threshold = 0.5*np.max(np.max(im))
    axis, distance = medial_axis((im>threshold), return_distance=True)
    distances = distance[np.nonzero(axis * distance)]
    thickness = distances*2

    h_range = [np.min(cell_coords[0]), np.max(cell_coords[0])]
    w_range = [np.min(cell_coords[1]), np.max(cell_coords[1])]
    return thickness, h_range, w_range


def generate_distortion_mask(im, sigma, num_centers=None):
    center = [im.shape[0]/2, im.shape[1]/2]
    area = im.shape[0]*im.shape[1]
    if num_centers is None:
        num_centers = int(20*area/300000)

    mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.float)
    x, y = np.mgrid[-center[0]: im.shape[0] - center[0], -center[1]: im.shape[1] - center[1]]

    for sig, num in zip(list(sigma),num_centers):
        for i in range(num):
            rand_sign = np.random.randint(-1,1)*2+1
            xpos = np.random.randint(low=-im.shape[0]/2, high=im.shape[0]/2)
            ypos = np.random.randint(low=-im.shape[1]/2, high=im.shape[1]/2)
            mask += rand_sign*np.exp(-((x-xpos) ** 2 / float(sig) + (y-ypos) ** 2 / float(sig)))

    max_min = np.max(mask) - np.min(mask)
    mask = (((mask- np.min(mask))/max_min)*255).astype(np.uint8)
    return mask

def custom_warp(im, landscape, power):
    # plt.imshow(im, cmap='gray');plt.show()

    der_w = cv2.Sobel(landscape, cv2.CV_32F, 1, 0, ksize=5)*power
    der_h = cv2.Sobel(landscape, cv2.CV_32F, 0, 1, ksize=5)*power
    # der_w = np.gradient(landscape, axis=1) * power
    # der_h = np.gradient(landscape, axis=0) * power

    # print(power)
    # plt.imshow(landscape, cmap='gray');plt.show()
    # plt.imshow(np.concatenate([np.expand_dims(der_w, axis=2),
    #                            np.expand_dims(der_h, axis=2),
    #                            np.expand_dims(np.zeros_like(der_w), axis=2)], axis=2), cmap='gray');plt.show()

    grid_h, grid_w = np.mgrid[0:im.shape[0],0:im.shape[1]]
    grid_sink = np.concatenate((np.expand_dims(der_h+grid_h,axis=2), np.expand_dims(der_w+grid_w,axis=2)),axis=2)
    map_x = np.append([], [ar[:, 1] for ar in grid_sink]).reshape(im.shape[0], im.shape[1])
    map_y = np.append([], [ar[:, 0] for ar in grid_sink]).reshape(im.shape[0], im.shape[1])
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    warped_image = cv2.remap(im, map_x_32, map_y_32, cv2.INTER_CUBIC)

    # plt.imshow(warped_image, cmap='gray');plt.show()
    # plt.subplot(311)
    # plt.imshow(der_w)
    # plt.subplot(312)
    # plt.imshow(im)
    # plt.subplot(313)
    # plt.imshow(warped_image)
    # plt.show()
    return warped_image


def preprocess_letter(im, target_thickness,
                      principal_rots, principal_scales,
                      rot, scale, hshear, wshear,
                      distortion_mask_sig, distortion_num_masks, distortion_power,
                      aa_scale=3.0,
                      verbose=False, display=False):
    ################### Upsample. Roughly standardize stroke thickness
    im_in = im.copy()
    im = rotate_n_scale(im, rotation=0, scale=[aa_scale, aa_scale], pad_val=0)

    thickness, _, _ = measure_obj_thickness(im)
    thickness = scipy.stats.mode(thickness)[0]
    thickness_delta = int(thickness - target_thickness*aa_scale)
    if thickness_delta >= 2:
        iters = int(np.round(thickness_delta/2))
        im = (scipy.ndimage.morphology.binary_erosion(im>128, iterations=iters) * 255).astype(np.uint8)
    elif thickness_delta <= -2:
        iters = int(np.round(-thickness_delta/2))
        im = (scipy.ndimage.morphology.binary_dilation(im>128, iterations=iters) * 255).astype(np.uint8)
    if display:
        im_std = im.copy()

    ################### Principally scale -> re-balance stroke thickness -> principally rotate
    # principally scale
    princ_scale_sign = np.random.randint(low=0, high=2)*2-1
    princ_scale_h_switch = np.random.randint(low=0, high=2)
    princ_scale_w_switch = np.random.randint(low=0, high=2)
    princ_scale_h_sampled = principal_scales[0][np.random.randint(low=0, high=len(principal_scales[0]))] ** princ_scale_h_switch
    princ_scale_w_sampled = principal_scales[1][np.random.randint(low=0, high=len(principal_scales[1]))] ** princ_scale_w_switch
    if (princ_scale_h_sampled < 0) or (princ_scale_h_sampled > 1) or (princ_scale_w_sampled < 0) or (princ_scale_w_sampled > 1):
        raise ValueError('principal scale must be between 0 and 1 ('+str(princ_scale_h_sampled)+')')
    else:
        princ_scale_h_sampled = princ_scale_h_sampled ** princ_scale_sign
        princ_scale_w_sampled = princ_scale_w_sampled ** princ_scale_sign
    im = rotate_n_scale(im, rotation=0, scale=[princ_scale_h_sampled, princ_scale_w_sampled], pad_val=0)
    # re-balance strokes
    h_struct = np.array([[False, False, False], [True, True, True], [False, False, False]])
    w_struct = np.array([[False, True, False], [False, True, False], [False, True, False]])
    if princ_scale_h_sampled < 1.:
        iters = int(np.round(target_thickness * aa_scale * (1 - princ_scale_h_sampled) / 2))
        im = (scipy.ndimage.morphology.binary_dilation(im, h_struct, iterations=iters) * 255.).astype(np.uint8)
    elif princ_scale_h_sampled > 1.:
        iters = int(np.round(target_thickness * aa_scale * (princ_scale_h_sampled - 1) / 2))
        im = (scipy.ndimage.morphology.binary_erosion(im, h_struct, iterations=iters) * 255).astype(np.uint8)
    if princ_scale_w_sampled < 1.:
        iters = int(np.round(target_thickness * aa_scale * (1- princ_scale_w_sampled) / 2))
        im = (scipy.ndimage.morphology.binary_dilation(im, w_struct, iterations=iters) * 255.).astype(np.uint8)
    elif princ_scale_w_sampled > 1.:
        iters = int(np.round(target_thickness * aa_scale * (princ_scale_w_sampled - 1) / 2))
        im = (scipy.ndimage.morphology.binary_erosion(im, w_struct, iterations=iters) * 255).astype(np.uint8)
    # principally rotate
    princ_rot_sign = np.random.randint(low=-1, high=2)  # principally rotate in two directions or ignore
    princ_rot_sampled = principal_rots[np.random.randint(low=0, high=len(principal_rots))]
    if (princ_rot_sampled < 0) or (princ_rot_sampled > 90):
        raise ValueError('principal rot must be between 0 and 90 degs')
    else:
        princ_rot_sampled *= princ_rot_sign
    princ_rot_sampled = np.arctan(np.tan(princ_rot_sampled*np.pi/180)*(princ_scale_h_sampled/princ_scale_w_sampled))*180/np.pi
    if display:
        im_principal = im.copy()
    im = rotate_n_scale(im, rotation=princ_rot_sampled, scale=[1.0, 1.0], pad_val=0)

    ################### Tight crop, rotation, scale, shear
    # semi-tight (30% slack) crop
    nonzero_coords = np.nonzero(im)
    if len(nonzero_coords[0]) == 0:
        raise ValueError('no live pixels. too much distortion?')
    obj_height = np.max(nonzero_coords[0]) - np.min(nonzero_coords[0])
    obj_width = np.max(nonzero_coords[1]) - np.min(nonzero_coords[1])
    major_axis = np.maximum(obj_height, obj_width)
    h_boundary_with_slack = [np.maximum(np.min(nonzero_coords[0]) - major_axis / 2, 0),
                             np.minimum(np.max(nonzero_coords[0]) + major_axis / 2, im.shape[0]-1)]
    w_boundary_with_slack = [np.maximum(np.min(nonzero_coords[1]) - major_axis / 2, 0),
                             np.minimum(np.max(nonzero_coords[1]) + major_axis / 2, im.shape[1]-1)]
    im = im[h_boundary_with_slack[0]:h_boundary_with_slack[1],
            w_boundary_with_slack[0]:w_boundary_with_slack[1]]
    # rot, scale
    im = rotate_n_scale(im, rotation=rot, scale=scale, pad_val=0)
    # shear
    im = shear(im, hshear=hshear, wshear=wshear, pad_val=0)
    if display:
        im_affine = im.copy()

    ################### Normalize stroke thickness + random distort
    thickness, _, _ = measure_obj_thickness(im)
    ######TODO: USE SOFT-MODE INSTEAD OF MEAN
    thickness_delta = np.mean(thickness) - target_thickness*aa_scale
    if thickness_delta >= 2:
        iters = int(np.round(thickness_delta/2))
        im = (scipy.ndimage.morphology.binary_erosion(im>128, iterations=iters) * 255).astype(np.uint8)
        if verbose:
            thickness_new, _, _ = measure_obj_thickness(im)
            print('eroded. thickness=' + str(int(np.mean(thickness_new))) + ', target=' + str(int(target_thickness*aa_scale)))
    elif thickness_delta <= -2:
        iters = int(np.round(-thickness_delta/2))
        im = (scipy.ndimage.morphology.binary_dilation(im>128, iterations=iters) * 255).astype(np.uint8)
        if verbose:
            thickness_new, _, _ = measure_obj_thickness(im)
            print('dilated. thickness=' + str(int(np.mean(thickness_new))) + ', target=' + str(int(target_thickness*aa_scale)))
    # random distortion
    landscape = generate_distortion_mask(im, sigma=np.array(distortion_mask_sig) * aa_scale,
                                         num_centers=distortion_num_masks)

    im = custom_warp(im, landscape, power=distortion_power * major_axis / 200)
    if display:
        im_warped = im.copy()

    ################### Downsample and tight crop
    im = rotate_n_scale(im, rotation=0, scale=[1/aa_scale, 1/aa_scale], pad_val=0)
    if verbose:
        thickness_new, _, _ = measure_obj_thickness(im)
        print('Final thickness=' + str(int(np.mean(thickness_new))) + ', target=' + str(int(target_thickness)))
    nonzero_coords = np.nonzero(im)
    if len(nonzero_coords[0]) == 0:
        raise ValueError('no live pixels. too much distortion?')
    im = im[np.min(nonzero_coords[0]):np.max(nonzero_coords[0]) + 1,
            np.min(nonzero_coords[1]):np.max(nonzero_coords[1]) + 1]

    if display:
        print('principal rotation=' + str(princ_rot_sampled))
        plt.figure(figsize=(16,3))
        plt.subplot(161);plt.imshow(im_in, cmap='gray');plt.title('in')
        plt.axis('off')
        plt.subplot(162);plt.imshow(im_std, cmap='gray');plt.title('standarized')
        plt.axis('off')
        plt.subplot(163);plt.imshow(im_principal, cmap='gray');plt.title('principal rot/scale')
        plt.axis('off')
        plt.subplot(164);plt.imshow(im_affine, cmap='gray');plt.title('affine')
        plt.axis('off')
        plt.subplot(165);plt.imshow(im_warped, cmap='gray');plt.title('warped')
        plt.axis('off')
        plt.subplot(166);plt.imshow(im, cmap='gray');plt.title('stroke-normalized')
        plt.axis('off')
        plt.show()

    ################### Find c of m
    nonzero_coords = np.nonzero(im)
    hcenter = int(np.round(np.mean(nonzero_coords[0])))
    wcenter = int(np.round(np.mean(nonzero_coords[1])))

    return im, [hcenter, wcenter]


# if __name__ == "__main__":
#     im = (255 - np.mean(scipy.misc.imread('/Users/junkyungkim/Desktop/aaa.jpg'), axis=2)).astype(np.float)
#     max_val = np.max(im)
#     im /= max_val
#     im_out = pixelate(im, patch_size=[45, 45], p_omission=0.15, sig_shift=20)
#     plt.subplot(121);
#     plt.imshow(im)
#     plt.subplot(122);
#     plt.imshow(im_out)
#     plt.show()
