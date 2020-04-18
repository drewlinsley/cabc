### python cluttered_nist_wrapper.py (N_MACHINES) (START_ID) (CURRENT_ID_FROM_1) (N_IMAGES)

import time
import sys
import numpy as np
import os

###########
# IDENTICAL TO CLUTTERED_NIST_WRAPPER
# PARAMS ARE SET FOR LARGER PIXELS AND GENERALLY LESS LOCAL FEATS
###########

class Args:
    def __init__(self, dataset_path, batch_id, n_images, segmentation_task,
                 positional_radius_range,
                 global_scale_pwr_sigma, global_rotation_sigma, global_shear_sigma,
                 letter_scale_pwr_sigma, letter_rotation_sigma, letter_shear_sigma,
                 save_images=True, save_seg=True, save_metadata=True, pause_display=False, verbose=False):

        # self.nist_path = nist_path
        self.dataset_path = dataset_path
        self.batch_id = batch_id
        self.n_images = n_images
        self.segmentation_task = segmentation_task

        self.positional_radius_range = positional_radius_range
        self.global_scale_pwr_sigma = global_scale_pwr_sigma
        self.global_rotation_sigma = global_rotation_sigma
        self.global_shear_sigma = global_shear_sigma
        self.letter_scale_pwr_sigma = letter_scale_pwr_sigma
        self.letter_rotation_sigma = letter_rotation_sigma
        self.letter_shear_sigma = letter_shear_sigma

        self.save_images = save_images
        self.save_seg = save_seg
        self.save_metadata = save_metadata
        self.pause_display = pause_display
        self.verbose = verbose

t = time.time()

import cluttered_nist
# import cluttered_nist2

## Fixed
font_root = '/Users/junkyungkim/Documents/PycharmProjects/cluttered_nist'
dataset_root = '/Users/junkyungkim/Desktop/cabc/'
num_machines = int(sys.argv[1])
current_id = int(sys.argv[2])
total_images = int(sys.argv[3])

if len(sys.argv)==4:
    print('Using default path...')
elif len(sys.argv)==5:
    print('Using custom save path...')
    dataset_root = str(sys.argv[4])
elif len(sys.argv)==6:
    print('Using custom font path...')
    dataset_root = str(sys.argv[4])
    font_root = str(sys.argv[5])
else:
    raise ValueError('wrong number of args')

args = Args(dataset_path=os.path.join(dataset_root),
            batch_id=current_id,
            n_images=total_images/num_machines,
            segmentation_task = False,
            positional_radius_range = [30, 35],
            global_scale_pwr_sigma = 1.,
            global_rotation_sigma = 20.,
            global_shear_sigma = 0.04,
            letter_scale_pwr_sigma = 1.,
            letter_rotation_sigma = 20.,
            letter_shear_sigma = 0.04)

args.font_root = font_root
args.font_names = ['FUTRFW.ttf',
                   'Instruction.otf']

args.segmentation_task = False
args.pixelate = True
args.verbose=False
args.verbose_final=True
args.display=False
args.display_final=False
args.aa_scale = 4.0
args.num_letters =2

# CONTROL
args.luminance_cue = False

args.std_font_sizes = np.array([113, 127, 127])*2 #np.array([113, 127, 127])*1.8
args.target_thickness = 7
args.marker_radius = 11 #9

args.letter_princ_scales = [[[1],[1]],[[1],[1]]]#[[[0.5, 0.6], [0.63, 0.88, 0.92]], [[0.5],[0.5]]]  # should come from a list
args.letter_princ_rotations = [[0],[0]]#[[21.5, 10.75], [16.5, 8.75]]  # should come from a list

args.distortion_mask_sig = [12000]
args.distortion_num_masks = [10]
args.distortion_power = 0.24 #0.18

args.positional_radius_range = [35, 80]
args.min_separation = 60

args.min_overlap = 0.01
args.max_overlap = 0.15

args.pixelate_patch_size = 5#3
args.pixelate_threshold = 0.3#0.15
args.pixelate_p_omission = 0.2#0.3
args.pixelate_jitter_sigma = 2

################################# DS: BASELINE
# args.global_scale_pwr_sigma = 0.22/np.sqrt(2)
# args.letter_scale_pwr_sigma = 0.22/np.sqrt(2)
# args.global_rotation_sigma = 8./np.sqrt(2)
# args.letter_rotation_sigma = 8./np.sqrt(2)
# args.global_shear_sigma = 0.08/np.sqrt(2)
# args.letter_shear_sigma = 0.08/np.sqrt(2)
args.global_scale_pwr_sigma = 0.5
args.letter_scale_pwr_sigma = 0.001
args.global_rotation_sigma = 30.
args.letter_rotation_sigma = 0.001
args.global_shear_sigma = 0.2
args.letter_shear_sigma = 0.001
args.positional_radius_range = [50, 60]
args.init_position_range = [-30, 30]
args.min_separation = 170
args.min_overlap = 0.01
args.max_overlap = 0.15
args.dataset_path = os.path.join(dataset_root,'baseline-')
cluttered_nist.from_wrapper(args)

# ################################# DS: ix1
# args.global_scale_pwr_sigma = 0.33/np.sqrt(2)
# args.letter_scale_pwr_sigma = 0.33/np.sqrt(2)
# args.global_rotation_sigma = 16./np.sqrt(2)
# args.letter_rotation_sigma = 16./np.sqrt(2)
# args.global_shear_sigma = 0.16/np.sqrt(2)
# args.letter_shear_sigma = 0.16/np.sqrt(2)
args.global_scale_pwr_sigma = 0.5/np.sqrt(2)
args.letter_scale_pwr_sigma = 0.5/np.sqrt(2)
args.global_rotation_sigma = 30./np.sqrt(2)
args.letter_rotation_sigma = 30./np.sqrt(2)
args.global_shear_sigma = 0.2/np.sqrt(2)
args.letter_shear_sigma = 0.2/np.sqrt(2)
args.positional_radius_range = [40, 70]
args.init_position_range = [-60, 60]
args.min_separation = 110
args.min_overlap = 0.01
args.max_overlap = 0.15
args.dataset_path = os.path.join(dataset_root,'ix1-')
cluttered_nist.from_wrapper(args)

# ################################# DS: ix2
# args.global_scale_pwr_sigma = 0.44/np.sqrt(2)
# args.letter_scale_pwr_sigma = 0.44/np.sqrt(2)
# args.global_rotation_sigma = 24./np.sqrt(2)
# args.letter_rotation_sigma = 24./np.sqrt(2)
# args.global_shear_sigma = 0.24/np.sqrt(2)
# args.letter_shear_sigma = 0.24/np.sqrt(2)
args.global_scale_pwr_sigma = 0.001
args.letter_scale_pwr_sigma = 0.5
args.global_rotation_sigma = 0.001
args.letter_rotation_sigma = 30.
args.global_shear_sigma = 0.001
args.letter_shear_sigma = 0.2
args.positional_radius_range = [30, 80]
args.init_position_range = None
args.min_separation = 60
args.min_overlap = 0.01
args.max_overlap = 0.15
args.dataset_path = os.path.join(dataset_root,'ix2')
cluttered_nist.from_wrapper(args)


# # ################################# DS: ix3
# args.dataset_path = os.path.join(dataset_root,'ix3')
# args.positional_radius_range = [11, 31]
# args.letter_scale_range = [0.7, 2.1]
# args.letter_rotation_range = [-36, 36]
# args.letter_shear_width = 0.24
# cluttered_nist.from_wrapper(args)

# args = Args(nist_path,
#             dataset_path=os.path.join(dataset_root),
#             batch_id=current_id,
#             n_images=total_images/num_machines,
#             positional_radius_range = [20, 22],
#             global_scale_range = [1.3, 1.5],
#             global_rotation_range = [-2, 2],
#             global_shear_width = 0.06,
#             letter_scale_range = [1.3, 1.5],
#             letter_rotation_range = [-1, 1],
#             letter_shear_width = 0.06)
# args.verbose=True
# # # ################################# DS: iy1
# args.dataset_path = os.path.join(dataset_root,'iy1')
# args.global_scale_range = [1.1, 1.7]
# args.global_rotation_range = [-12, 12]
# args.global_shear_width = 0.12
# cluttered_nist.from_wrapper(args)
# #
# # # ################################# DS: iy1
# args.dataset_path = os.path.join(dataset_root,'iy2')
# args.global_scale_range = [0.9, 1.9]
# args.global_rotation_range = [-24, 24]
# args.global_shear_width = 0.18
# cluttered_nist.from_wrapper(args)
# #
# # # ################################# DS: iy1
# args.dataset_path = os.path.join(dataset_root,'iy3')
# args.global_scale_range = [0.7, 2.1]
# args.global_rotation_range = [-36, 36]
# args.global_shear_width = 0.24
# cluttered_nist.from_wrapper(args)

elapsed = time.time() - t
print('ELAPSED TIME OVER ALL CONDITIONS : ', str(elapsed))
