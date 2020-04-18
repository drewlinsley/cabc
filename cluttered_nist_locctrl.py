import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import scipy
import scipy.stats
import cv2
import os
import emnist_helpers

def save_metadata(metadata, contour_path, batch_id):
    # Converts metadata (list of lists) into an nparray, and then saves
    metadata_path = os.path.join(contour_path, 'metadata')
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
    metadata_fn = str(batch_id) + '.npy'
    np.save(os.path.join(metadata_path,metadata_fn), metadata)

# Accumulate metadata
def accumulate_meta(array, im_subpath, seg_sub_path, im_filename, nimg,
                    image_category, letter_img_indices):

    # NEW VERSION
    array += [[im_subpath, seg_sub_path, im_filename, nimg, image_category] + letter_img_indices]
    return array

# Accumulate metadata
def accumulate_meta_segment(array, im_subpath, seg_sub_path, im_filename, nimg,
                            letter_img_indices):

    # NEW VERSION
    array += [[im_subpath, seg_sub_path, im_filename, nimg] + letter_img_indices]
    return array

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def translate_coord(coord, orientation, dist, allow_float=False):
    y_displacement = float(dist)*np.sin(orientation)
    x_displacement = float(dist)*np.cos(orientation)
    if allow_float is True:
        new_coord = [coord[0]+y_displacement, coord[1]+x_displacement]
    else:
        new_coord = [int(np.ceil(coord[0] + y_displacement)), int(np.ceil(coord[1] + x_displacement))]
    return new_coord

def get_availability_notouch(im, com_on_im, radius, canvas_size,
                             existing_canvas=None, existing_deg=None,
                             min_separation_deg=45, min_separation_px=15):
    if (min_separation_deg>180):
        return ValueError('min_separation_deg should be leq than 180')
    # Filter available positions to prevent overlap
    if existing_canvas is not None:
        print('placing second letter')
        im_inverted = im[::-1,::-1]
        com_on_im_inverted = [im.shape[0]-com_on_im[0]-1, im.shape[1]-com_on_im[1]-1]
        h_offset = com_on_im[0]-com_on_im_inverted[0]
        w_offset = com_on_im[1]-com_on_im_inverted[1]
        pad_thickness = ((np.maximum(h_offset, 0) + min_separation_px, np.maximum(-h_offset, 0) + min_separation_px),
                         (np.maximum(w_offset, 0) + min_separation_px, np.maximum(-w_offset, 0) + min_separation_px))
        im_inverted_padded = \
            np.pad(im_inverted, pad_thickness, 'constant', constant_values=((0,0), (0,0))).astype(np.bool)
        im_inverted_dilated = scipy.ndimage.morphology.binary_dilation(im_inverted_padded, iterations=min_separation_px)
        occupation_mask = cv2.dilate(existing_canvas.astype(np.uint8), im_inverted_dilated.astype(np.uint8)).astype(np.bool)
        # print('positive = ' + str(np.sum(occupation_mask.astype(np.int)[:])))
        # print('negative = ' + str(np.sum(1 - occupation_mask.astype(np.int)[:])))
    else:
        print('placing first letter')
        occupation_mask = np.zeros((canvas_size[0], canvas_size[1])).astype(np.bool)

    # Filter available positions to ensure angular dist between COMs
    availability_mask = np.zeros((canvas_size[0], canvas_size[1]))
    canvase_center = [canvas_size[0]/2, canvas_size[1]/2]
    if existing_deg is not None:
        degs = [x for x in range(360) if np.abs(existing_deg-x)]
    else:
        degs = [x for x in range(360)]
    for deg in degs:
        coord = translate_coord(canvase_center, deg*np.pi/180, radius, allow_float=False)
        availability_mask[coord[0], coord[1]] = not occupation_mask[coord[0], coord[1]]

    # Filter available positions to prevent overflow of obj window
    availability_mask[:com_on_im[0]+1, :] = False
    availability_mask[-(im.shape[0] - com_on_im[0])-1:, :] = False
    availability_mask[:, :com_on_im[1]+1] = False
    availability_mask[:, -(im.shape[1] - com_on_im[1])-1:] = False

    # if existing_canvas is not None:
    #     plt.subplot(141);plt.imshow(im)
    #     plt.subplot(142);plt.imshow(existing_canvas.astype(np.uint8))
    #     plt.subplot(143);plt.imshow(occupation_mask)
    #     plt.subplot(144);plt.imshow(availability_mask.astype(np.uint8))
    #     plt.show()
    return availability_mask

def place_on_canvas(im, com_in_im, canvas_size, com_in_canvas):
    canvas = np.zeros((canvas_size[0], canvas_size[1]))
    hrange = [com_in_canvas[0] - com_in_im[0], com_in_canvas[0] + (im.shape[0] - com_in_im[0])]
    wrange = [com_in_canvas[1] - com_in_im[1], com_in_canvas[1] + (im.shape[1] - com_in_im[1])]
    # if canvas[hrange[0]:hrange[1], wrange[0]:wrange[1]].shape != im.shape:
    #     import ipdb
    #     ipdb.set_trace()
    canvas[hrange[0]:hrange[1], wrange[0]:wrange[1]] = im
    return canvas

def gauss_mask(shape=(10,10),sigma=4):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def obj_exclusive_mask(master_canvas, dilate_others=0, obj_idx=0, dynamic_range=255.):
    temp_master_canvas = master_canvas.copy()*dynamic_range/np.max(master_canvas)
    if dilate_others>0:
        for iletter in range(temp_master_canvas.shape[2]):
            if iletter != obj_idx:
                temp_master_canvas[:,:,iletter] = (scipy.ndimage.morphology.binary_dilation(temp_master_canvas[:,:,iletter] > 128, iterations=dilate_others)*dynamic_range).astype(np.uint8)
    # plt.imshow(temp_master_canvas[:, :, 0]);
    # plt.show()
    # import ipdb
    # ipdb.set_trace()
    obj_mask = np.copy(temp_master_canvas[:,:,obj_idx])>dynamic_range/3
    temp_master_canvas[:, :, obj_idx] = 0
    remainder = np.max(temp_master_canvas, axis=2)>dynamic_range/3
    return ((obj_mask & ~remainder)*dynamic_range).astype(np.uint8)

def get_circle_mask(origin, radius, imsize):
    mask = np.zeros((imsize[0], imsize[1]))
    y, x = np.ogrid[-origin[0]:(imsize[0]-origin[0]),
                    -origin[1]:(imsize[1]-origin[1])]
    inside = x ** 2 + y ** 2 <= (radius+1)**2
    outside = x ** 2 + y ** 2 >= (radius-1)**2
    mask[inside & outside] = 1

    return mask.astype(np.int)

def sample_coord_from_mask(mask, return_deg=False):
    normalized_mask = mask.astype(np.float)/np.sum(mask[:])
    center = [mask.shape[0]/2, mask.shape[1]/2]
    # sample a coordinate using cmf method
    nonzero_coords = np.nonzero(normalized_mask)
    pmf = normalized_mask[nonzero_coords]
    cmf = np.dot(pmf, np.triu(np.ones((pmf.shape[0], pmf.shape[0])), 0))
    sampled_index = np.argmax(cmf - np.random.rand() >= 0)
    sampled_coord = [nonzero_coords[0][sampled_index], nonzero_coords[1][sampled_index]]
    if return_deg:
        y_displacement = sampled_coord[0] - center[0]
        x_displacement = sampled_coord[1] - center[1]
        if x_displacement==0:
            sampled_deg = 90*np.sign(y_displacement)
        else:
            sampled_deg = np.arctan(float(y_displacement)/x_displacement)
        return sampled_coord, sampled_deg
    else:
        return sampled_coord

def draw_circle(image, coordinate, radius, aa_scale, dynamic_range=255.):
    y, x = np.ogrid[-coordinate[0]*aa_scale:(image.shape[0]-coordinate[0])*aa_scale,
                    -coordinate[1]*aa_scale:(image.shape[1]-coordinate[1])*aa_scale]
    white_mask = scipy.misc.imresize((x ** 2 + y ** 2 <= ((radius * aa_scale) ** 2)).astype(np.float)*dynamic_range,
                                     (image.shape[0], image.shape[1]), interp='lanczos')
    black_mask = scipy.misc.imresize((x ** 2 + y ** 2 > (((radius - 3)* aa_scale) ** 2)).astype(np.float)*dynamic_range,
                                     (image.shape[0], image.shape[1]), interp='lanczos')
    max_val = np.max(image)
    circle = np.minimum(np.maximum(image,white_mask*max_val/255), black_mask*max_val/255)
    return circle

def one_image(positive_or_negative, num_letters, segmentation_task, marker_radius,
              font_root, font_names, std_font_sizes, target_thickness,
              pixelate_patch_size, pixelate_threshold, pixelate_p_omission, pixelate_jitter_sigma,
              letter_princ_scales, letter_princ_rotations,
              letter_rand_scale, letter_rand_rotation, letter_rand_shear,
              letter_distortion_mask_sig, letter_distortion_num_masks, letter_distortion_power,
              positional_radius, min_separation, min_overlap, max_overlap,
              positional_radius_range=None,
              pixelate=False, aa_scale=3.0, verbose=False, verbose_final=False, display=False, display_final=False,
              luminance_cue=False):

    ############## PARAMETERS (FIXED)
    canvas_size = [350, 350]
    combination_mode = 0 # 0 iid uniform 1 always different 2 always same

    ############## SAMPLE FONT
    font_idx = np.random.randint(low=0, high=len(font_names))
    fontname = font_names[font_idx]
    fontsize = int(std_font_sizes[font_idx])

    ############## INITIALIZE
    min_overlap_check = False
    max_overlap_check = False
    num_failure = 0
    category = np.random.randint(low=0, high=26)
    master_canvas = np.zeros((canvas_size[0], canvas_size[1], num_letters))
    master_canvas_mask = np.zeros((canvas_size[0], canvas_size[1], num_letters))
    if display_final:
        master_canavs_raw = []
    num_letters_drawn = 0
    letter_imgs = []
    letter_coms = []
    letter_img_indices = []

    while num_letters_drawn<num_letters:
        ############## SAMPLE CATEGORY
        if (combination_mode==0) | (combination_mode==1):
            new_category = np.random.randint(low=0, high=26)
            while (combination_mode==1) & (category==new_category):
                new_category = np.random.randint(low=0, high=26)
            category=new_category

        ############## SAMPLE IMAGE (FONT)
        font = ImageFont.truetype(os.path.join(font_root, 'fonts', fontname), fontsize)
        ascii_category = category + 65
        raw_im = Image.new("L", (int(fontsize*2), int(fontsize*2)), (0))
        draw = ImageDraw.Draw(raw_im)
        draw.text((0, 0), chr(ascii_category), font=font, fill=(255))
        ImageDraw.Draw(raw_im)
        raw_im = np.array(raw_im).astype(np.float)
        nonzero_coords = np.nonzero(raw_im)
        raw_im = raw_im[np.min(nonzero_coords[0]):np.max(nonzero_coords[0]) + 1,
                        np.min(nonzero_coords[1]):np.max(nonzero_coords[1]) + 1]
        raw_im = np.pad(raw_im, ((int(100*aa_scale), int(100*aa_scale)), (int(100*aa_scale), int(100*aa_scale))), mode='constant', constant_values=((0, 0), (0, 0)))

        ############## PREPROCESS LETTER
        import preprocess
        im, com_in_crop = preprocess.preprocess_letter(raw_im, target_thickness,
                                                       letter_princ_rotations[font_idx], letter_princ_scales[font_idx],
                                                       letter_rand_rotation[num_letters_drawn], letter_rand_scale[num_letters_drawn],
                                                       letter_rand_shear[num_letters_drawn][0], letter_rand_shear[num_letters_drawn][1],
                                                       letter_distortion_mask_sig, letter_distortion_num_masks, letter_distortion_power,
                                                       aa_scale=aa_scale, verbose=verbose, display=display)
        letter_imgs.append(im)
        letter_coms.append(com_in_crop)
        letter_img_indices.append(category)
        num_letters_drawn += 1

    ############## PLACE ON A CANVAS
    success = False
    while (not success):
        temp_canvas=[]
        existing_deg = None
        canvas = None
        for i, letter_im in enumerate(letter_imgs):
            availability_mask = \
            get_availability_notouch(letter_im, letter_coms[i], positional_radius, canvas_size,
                                     existing_canvas=canvas, existing_deg=existing_deg,
                                     min_separation_deg=min_separation, min_separation_px=25)
            if np.any(availability_mask)==False:
                print('No available space to place.' + str(num_failure))
                num_failure += 1
                break
            else:
                sampled_coord, sampled_deg = sample_coord_from_mask(availability_mask, return_deg=True)
                canvas = place_on_canvas(letter_im, letter_coms[i], canvas_size, sampled_coord)
                temp_canvas.append(canvas)
                existing_deg = sampled_deg
                success = True
        if not success:
            if num_failure > 10:
                if verbose_final:
                    print('10 failed attempts due to space availability.')
                return None, None, None, False
        else:
            for i, im in enumerate(temp_canvas):
                master_canvas[:, :, i] = im
                master_canvas_mask[:,:,i] = (scipy.ndimage.morphology.binary_dilation(im>128, iterations=int(target_thickness*1.4)) * 255).astype(np.uint8)
                if display_final:
                    master_canavs_raw.append(im.copy())

    ############## PIXELATE
    if pixelate:
        master_canvas = preprocess.pixelate_obj(master_canvas, [pixelate_patch_size,pixelate_patch_size], pixelate_threshold,
                                               pixelate_p_omission, pixelate_jitter_sigma, jitter_type='gaussian', ignore_patch_fit=True)

    fused_img = np.max(master_canvas, axis=2)

    ################################################# DRAW MARKERS
    markers=[]
    # use gaussian distribution to sample the first marker (makes it efficient by minimizing rejection probability)
    mask = gauss_mask(shape=canvas_size, sigma=canvas_size[0]/3)
    self_exclusive_mask = obj_exclusive_mask(master_canvas, dilate_others=int(target_thickness*2), obj_idx=0, dynamic_range=255.)
    rest_exclusive_mask = np.zeros_like(self_exclusive_mask)
    for i_obj in range(1,num_letters):
        rest_exclusive_mask = np.maximum(rest_exclusive_mask, obj_exclusive_mask(master_canvas, dilate_others=int(target_thickness*2), obj_idx=i_obj, dynamic_range=255.))
    marker_mask = mask*self_exclusive_mask
    # sample a coordinate using cmf method
    markers.append(sample_coord_from_mask(marker_mask))
    fused_img = draw_circle(fused_img, markers[0], marker_radius, aa_scale=aa_scale)
    # sample second coordinate from a fixed dist from the first marker
    if not segmentation_task:
        max_radius = np.max([markers[0][0], markers[0][1], canvas_size[0]-1-markers[0][0], canvas_size[1]-1-markers[0][1]])
        max_radius = np.min([max_radius, positional_radius*3]) # cap marker rad by letter distance*2
        success = False
        num_failure = 0
        while not success:
            sampled_radius = np.random.randint(low=marker_radius*2+1, high=max_radius)
            circle_mask = get_circle_mask(markers[0], sampled_radius, canvas_size)
            overlap_with_self = circle_mask & self_exclusive_mask.astype(np.bool)
            overlap_with_others = circle_mask & rest_exclusive_mask.astype(np.bool)
            if (np.sum(overlap_with_self[:])>0) & (np.sum(overlap_with_others[:])>0):
                success = True
                if positive_or_negative==0:
                    markers.append(sample_coord_from_mask(overlap_with_others))
                elif positive_or_negative==1:
                    markers.append(sample_coord_from_mask(overlap_with_self))
            else:
                num_failure += 1
                if verbose_final:
                    string = 'marker?' + str(num_failure)
                    print(string)
                if num_failure >= 10:
                    if verbose_final:
                        print('10 failed attempts')
                    return None, None, None, False
        fused_img = draw_circle(fused_img, markers[1], marker_radius, aa_scale=aa_scale)

    if display_final:
        for i_letter in range(num_letters):
            plt.subplot(4, num_letters, i_letter+1);plt.imshow(letter_imgs[i_letter]);plt.title(fontname)
            plt.subplot(4, num_letters, num_letters+i_letter+1);plt.imshow(master_canavs_raw[i_letter])
            plt.subplot(4, num_letters, num_letters*2+i_letter+1);plt.imshow(master_canvas[:,:,i_letter])
            plt.subplot(4, num_letters, num_letters*3+2);plt.imshow(fused_img)
        plt.show()
    return fused_img, master_canvas, letter_img_indices, True

#############################START HERE
def from_wrapper(args):
    import time
    t_alpha = time.time()
    iimg = 0
    failed_imgs = 0

    # ims_list, num_ims_list = emnist_helpers.load_list(os.path.join(args.nist_path,'list.npy'))

    if (args.save_images):
        dataset_sub_path = os.path.join('imgs', str(args.batch_id))
        if not os.path.exists(os.path.join(args.dataset_path, dataset_sub_path)):
            os.makedirs(os.path.join(args.dataset_path, dataset_sub_path))
    if (args.save_seg):
        seg_sub_path = os.path.join('segs', str(args.batch_id))
        if not os.path.exists(os.path.join(args.dataset_path, seg_sub_path)):
            os.makedirs(os.path.join(args.dataset_path, seg_sub_path))
    if args.save_metadata:
        metadata = []
        # CHECK IF METADATA FILE ALREADY EXISTS
        metadata_path = os.path.join(args.dataset_path, 'metadata')
        if not os.path.exists(metadata_path):
            os.makedirs(metadata_path)
        metadata_fn = str(args.batch_id) + '.npy'
        metadata_full = os.path.join(metadata_path, metadata_fn)
        if os.path.exists(metadata_full):
            print('Metadata file already exists.')
            return

    while (iimg < args.n_images):
        t0 = time.time()
        print('Image# : %s' % (iimg))
        im_fn = "sample_%s.png" % (iimg)

        ######################### SAMPLE RANDOM PARAMS
        positive_or_negative = 1 if np.random.rand() > 0.5 else 0
        positional_radius = args.positional_radius_range[0] + np.random.rand()*(args.positional_radius_range[1] - args.positional_radius_range[0])
        # global random scale (composition is multiplicative)
        global_scale_sign = 1. if np.random.rand() > 0.5 else -1.
        global_scale_h_power = global_scale_sign* np.abs(scipy.stats.truncnorm(-2, 2, loc=0, scale=args.global_scale_pwr_sigma).rvs())
        global_scale_w_power = global_scale_sign* np.abs(scipy.stats.truncnorm(-2, 2, loc=0, scale=args.global_scale_pwr_sigma).rvs())
        global_scale = [0.5 ** global_scale_h_power, 0.5 ** global_scale_w_power]
        # global random rotation (composition is linear)
        global_rotation = scipy.stats.truncnorm(-2, 2, loc=0, scale=args.global_rotation_sigma).rvs()
        # global random shear (composition is linear)
        global_shear_sign = 1. if np.random.rand() > 0.5 else -1.
        global_shear_h = global_shear_sign* np.abs(scipy.stats.truncnorm(-2, 2, loc=0, scale=args.global_shear_sigma).rvs())
        global_shear_w = global_shear_sign* np.abs(scipy.stats.truncnorm(-2, 2, loc=0, scale=args.global_shear_sigma).rvs())
        global_shear = [global_shear_h, global_shear_w]
        # letter random scale
        letter_scale = []
        letter_rotation = []
        letter_shear = []
        for i in range(args.num_letters):
            # combine global scale/rot/shear with letter
            # letter random scale
            letter_scale_sign = 1. if np.random.rand() > 0.5 else -1.
            letter_scale_h_power = letter_scale_sign* np.abs(scipy.stats.truncnorm(-2, 2, loc=0, scale=args.letter_scale_pwr_sigma).rvs())
            letter_scale_w_power = letter_scale_sign* np.abs(scipy.stats.truncnorm(-2, 2, loc=0, scale=args.letter_scale_pwr_sigma).rvs())
            letter_scale.append([(0.5 ** letter_scale_h_power)*global_scale[0], (0.5 ** letter_scale_w_power)*global_scale[1]])
            # letter random rotation
            letter_rotation.append(scipy.stats.truncnorm(-2, 2, loc=0, scale=args.letter_rotation_sigma).rvs() + global_rotation)
            # letter random shear
            letter_shear_sign = 1. if np.random.rand() > 0.5 else -1.
            letter_shear_h = letter_shear_sign * np.abs(scipy.stats.truncnorm(-2, 2, loc=0,scale=args.letter_shear_sigma).rvs())
            letter_shear_w = letter_shear_sign * np.abs(scipy.stats.truncnorm(-2, 2, loc=0,scale=args.letter_shear_sigma).rvs())
            letter_shear.append([letter_shear_h + global_shear[0], letter_shear_w + global_shear[1]])

        ######################### GENERATE IMAGE
        if args.verbose_final:
            print('global_scale= %.2f, %.2f'%(global_scale[0], global_scale[1]))
            print('global_rot= %.2f'%(global_rotation))
            print('global_shear= %.2f, %.2f' % (global_shear[0], global_shear[1]))
            print('letter1_scale= %.2f, %.2f'%(letter_scale[0][0]/global_scale[0], letter_scale[0][1]/global_scale[1]))
            print('letter1_rot= %.2f'%(letter_rotation[0]-global_rotation))
            print('letter1_shear= %.2f, %.2f' % (letter_shear[0][0] - global_shear[0], letter_shear[0][1] - global_shear[1]))
            print('letter2_scale= %.2f, %.2f'%(letter_scale[1][0]/global_scale[0], letter_scale[1][1]/global_scale[1]))
            print('letter2_rot= %.2f'%(letter_rotation[1]-global_rotation))
            print('letter2_shear= %.2f, %.2f' % (letter_shear[1][0] - global_shear[0], letter_shear[1][1] - global_shear[1]))
        img, inst_seg, letter_img_indices, success = \
            one_image(positive_or_negative, args.num_letters, args.segmentation_task, args.marker_radius,
                  args.font_root, args.font_names, args.std_font_sizes, args.target_thickness,
                  args.pixelate_patch_size, args.pixelate_threshold, args.pixelate_p_omission, args.pixelate_jitter_sigma,
                  args.letter_princ_scales, args.letter_princ_rotations,
                  letter_scale, letter_rotation, letter_shear,
                  args.distortion_mask_sig, args.distortion_num_masks, args.distortion_power,
                  positional_radius, args.min_separation, args.min_overlap, args.max_overlap,
                  positional_radius_range=args.positional_radius_range,
                  pixelate=args.pixelate, aa_scale=args.aa_scale,
                    verbose=args.verbose, verbose_final=args.verbose_final, display=args.display, display_final=args.display_final)

        if success:
            if (args.save_images):
                scipy.misc.imsave(os.path.join(args.dataset_path, dataset_sub_path, im_fn), img)
            if (args.save_seg):
                if args.num_letters==2:
                    inst_seg=np.concatenate([inst_seg, np.zeros((350, 350, 1))], axis=2)
                scipy.misc.imsave(os.path.join(args.dataset_path, seg_sub_path, im_fn), inst_seg)
            if (args.save_metadata):
                if args.segmentation_task:
                    metadata = accumulate_meta_segment(metadata,
                                               dataset_sub_path, seg_sub_path, im_fn, iimg,
                                               letter_img_indices)
                else:
                    metadata = accumulate_meta(metadata,
                                               dataset_sub_path, seg_sub_path, im_fn, iimg,
                                               positive_or_negative, letter_img_indices)
            elapsed = time.time() - t0
            print('PER IMAGE : ', str(elapsed))
            iimg += 1
        else:
            #############################
            failed_imgs += 1
            continue
    if (args.save_metadata):
        matadata_nparray = np.array(metadata)
        save_metadata(matadata_nparray, args.dataset_path, args.batch_id)
    print('TOTAL GENERATED IMGS = '+ str(iimg))
    print('TOTAL FAILED IMGS = ' + str(failed_imgs))
    print('TOTAL ELAPSED TIME = ' + str(time.time() - t_alpha))
    return

if __name__ == "__main__":
    # COMPILE LIST OF LETTER IMAGES
    nist_path = '/Users/junkyungkim/Desktop/by_class/'
    ims_list, num_ims_list = emnist_helpers.load_list(os.path.join(nist_path,'list.npy'))
    _ = one_image(ims_list, num_ims_list)