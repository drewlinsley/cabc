import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import scipy
import scipy.stats
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

def place_on_canvas(im, com_in_im, canvas_size, com_in_canvas):
    canvas = np.zeros((canvas_size[0],canvas_size[1]))
    hrange = [com_in_canvas[0] - com_in_im[0], com_in_canvas[0] + (im.shape[0] - com_in_im[0])]
    wrange = [com_in_canvas[1] - com_in_im[1], com_in_canvas[1] + (im.shape[1] - com_in_im[1])]
    if (hrange[0]<0) | (hrange[1]>=canvas_size[0]) | (wrange[0]<0) | (wrange[1]>=canvas_size[1]):
        return canvas, False
    canvas[hrange[0]:hrange[1], wrange[0]:wrange[1]] = im
    return canvas, True

def translate_coord(coord, orientation, dist, allow_float=False):
    y_displacement = float(dist)*np.sin(orientation)
    x_displacement = float(dist)*np.cos(orientation)
    if allow_float is True:
        new_coord = [coord[0]+y_displacement, coord[1]+x_displacement]
    else:
        new_coord = [int(np.ceil(coord[0] + y_displacement)), int(np.ceil(coord[1] + x_displacement))]
    return new_coord

def sample_positions_circle(radius, canvas_size, num_letters,
                            positional_radius_range=None, min_separation=45):
    if (min_separation>180):
        return ValueError('min_separation should be leq than 180')
    occupancy=np.ones(360)
    canvase_center = [canvas_size[0]/2, canvas_size[1]/2]
    degrees_list = []
    for iletter in range(num_letters):
        available_degrees = np.nonzero(occupancy)[0]
        if available_degrees.shape[0] == 0:
            print('no position available')
            return
        if (iletter == 0) & (positional_radius_range is not None):
            degree = np.random.randint(low = positional_radius_range[0], high=positional_radius_range[1]+1)
            if degree<0:
                degree = 360-degree
        else:
            degree_idx = np.random.randint(low=0, high=available_degrees.shape[0])
            degree = available_degrees[degree_idx]
        degrees_list.append(degree)
        if degree - min_separation < 0:
            occupancy[:degree + min_separation + 1] = 0
            occupancy[360 + degree - min_separation:] = 0
        elif degree + min_separation + 1 > 360:
            occupancy[degree - min_separation:] = 0
            occupancy[:degree + min_separation + 1 - 360] = 0
        else:
            occupancy[degree-min_separation:degree+min_separation+1] = 0
    positions_list = []
    for degree in degrees_list:
        coord = translate_coord(canvase_center, degree*np.pi/180, radius, allow_float=False)
        positions_list.append(coord)
    return positions_list, degrees_list

def check_overlaps(master_canvas, min_overlap, max_overlap):
    min_overlap_check = False
    max_overlap_check = True

    sum = np.sum(master_canvas,axis=2)
    for icanvas in range(master_canvas.shape[2]):
        mask = (master_canvas[:,:,icanvas]>128).astype(np.int)
        overlap_mask = (sum*mask>300).astype(np.int)
        area_overlap = np.sum(overlap_mask[:])
        area_letter = np.sum(mask[:])
        # print(float(area_overlap)/ area_letter)
        # print('min_overlap='+str(min_overlap))
        # import ipdb
        # ipdb.set_trace()
        if float(area_overlap)/ area_letter >= min_overlap:
            min_overlap_check = True
        if float(area_overlap)/ area_letter >= max_overlap:
            max_overlap_check = False
    return min_overlap_check, max_overlap_check, float(area_overlap)/ area_letter

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

def sample_coord_from_mask(mask):
    if np.sum(mask[:])==0:
        return ValueError('mask should have at least one nonzero element')
    normalized_mask = mask.astype(np.float)/np.sum(mask[:])

    # sample a coordinate using cmf method
    nonzero_coords = np.nonzero(normalized_mask)
    pmf = normalized_mask[nonzero_coords]
    cmf = np.dot(pmf, np.triu(np.ones((pmf.shape[0], pmf.shape[0])), 0))
    sampled_index = np.argmax(cmf - np.random.rand() >= 0)
    sampled_coord = [nonzero_coords[0][sampled_index], nonzero_coords[1][sampled_index]]
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

        ############## PREPROCESS & PLACE LETTER
        import preprocess
        im, com_in_crop = preprocess.preprocess_letter(raw_im, target_thickness,
                                                       letter_princ_rotations[font_idx], letter_princ_scales[font_idx],
                                                       letter_rand_rotation[num_letters_drawn], letter_rand_scale[num_letters_drawn],
                                                       letter_rand_shear[num_letters_drawn][0], letter_rand_shear[num_letters_drawn][1],
                                                       letter_distortion_mask_sig, letter_distortion_num_masks, letter_distortion_power,
                                                       aa_scale=aa_scale, verbose=verbose, display=display)
        letter_imgs.append(im)
        letter_img_indices.append(category)
        num_letters_drawn += 1

    ############## PLACE ON A CANVAS
    while (not min_overlap_check) | (not max_overlap_check):
        temp_canvas=[]
        com_in_canvas, _ = sample_positions_circle(positional_radius, canvas_size, num_letters,
                                                   positional_radius_range=positional_radius_range, min_separation=min_separation)
        for i, im in enumerate(letter_imgs):
            canvas, success = place_on_canvas(im, com_in_crop, canvas_size, com_in_canvas[i])
            if not success:
                break
            else:
                temp_canvas.append(canvas)
        if not success:
            if verbose_final:
                print('Frame overflow. Letter might be too large / canvas too small' + str(num_failure))
            num_failure += 1
            if num_failure > 10:
                if verbose_final:
                    print('10 failed attempts due to frame overflow.')
                return None, None, None, False
            else:
                continue
        else:
            for i, im in enumerate(temp_canvas):
                master_canvas[:, :, i] = im
                master_canvas_mask[:,:,i] = (scipy.ndimage.morphology.binary_dilation(im>128, iterations=int(target_thickness*1.4)) * 255).astype(np.uint8)
                if display_final:
                    master_canavs_raw.append(im.copy())

        ############## CHECK FOR OVERLAPS
        min_overlap_check, max_overlap_check, overlap = check_overlaps(master_canvas_mask, min_overlap, max_overlap)
        string = 'overlap=%.2f' % (overlap)
        if (not min_overlap_check) | (not max_overlap_check):
            num_failure += 1
            if verbose_final:
                if not min_overlap_check:
                    string += (' too small?' + str(num_failure))
                if not max_overlap_check:
                    string += (' too much?' + str(num_failure))
                print(string)
            if num_failure >= 10:
                if verbose_final:
                    print('10 failed attempts due to overlap. Restarting image.')
                return None, None, None, False
            else:
                continue
        else:
            if verbose_final:
                print('overlap=%.2f' % (overlap))

    ############## PIXELATE
    if pixelate:
        master_canvas = preprocess.pixelate_obj(master_canvas, [pixelate_patch_size,pixelate_patch_size], pixelate_threshold,
                                               pixelate_p_omission, pixelate_jitter_sigma, jitter_type='gaussian', ignore_patch_fit=True)

    fused_img = np.max(master_canvas, axis=2)
    if luminance_cue:
        master_canvas.astype(np.int)
        min_difference = 40
        lum1 = np.random.randint(low=128, high=256)
        lum2 = lum1
        while (lum2 - lum1)**2 < min_difference**2:
            lum2 = np.random.randint(low=128, high=256)
        master_canvas[:, :, 0] = master_canvas[:, :, 0] * lum1 / np.max(master_canvas[:, :, 0])
        master_canvas[:, :, 1] = master_canvas[:, :, 1] * lum2 / np.max(master_canvas[:, :, 1])
        fused_img = master_canvas[:, :, 0] + master_canvas[:, :, 1]
    # plt.subplot(131);plt.imshow(master_canvas[:, :, 0])
    # plt.subplot(132);plt.imshow(master_canvas[:, :, 1])
    # plt.subplot(133);plt.imshow(fused_img)
    # plt.show()

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
    else:
        master_canvas[:,:,0] = draw_circle(master_canvas[:,:,0], markers[0], marker_radius, aa_scale=aa_scale)

    if display_final:
        for i_letter in range(num_letters):
            plt.subplot(4, num_letters, i_letter+1);plt.imshow(letter_imgs[i_letter], cmap='gray');plt.title(fontname)
            plt.axis('off')
            plt.subplot(4, num_letters, num_letters+i_letter+1);plt.imshow(master_canavs_raw[i_letter], cmap='gray')
            plt.axis('off')
            plt.subplot(4, num_letters, num_letters*2+i_letter+1);plt.imshow(master_canvas[:,:,i_letter], cmap='gray')
            plt.axis('off')
            plt.subplot(4, num_letters, num_letters*3+2);plt.imshow(fused_img, cmap='gray')
            plt.axis('off')
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
                  positional_radius_range = args.positional_radius_range,
                  pixelate=args.pixelate, aa_scale=args.aa_scale, verbose=args.verbose, verbose_final=args.verbose_final, display=args.display, display_final=args.display_final,
                  luminance_cue=args.luminance_cue)

        if success:
            if (args.save_images):
                scipy.misc.imsave(os.path.join(args.dataset_path, dataset_sub_path, im_fn), img)
            if (args.save_seg):
                if args.segmentation_task:
                    inst_seg = inst_seg[:,:,0]
                else:
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