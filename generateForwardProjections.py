import cv2
import os
import time
import cupy as cp
import numpy as np
from PIL import Image
import generateBackProjections as bp

def compile(img_np, blank_path, n_proj_vert, n_proj_horz, downsample_ratio, proj_inplane_offset, proj_outplane_offset,
                        proj_width, proj_height, trap_path, path, mode, img_path_processed, img_path_final):
    compiled_tests_blank = []
    compiled_tests = []

    print(f'Adjusting image...', end=' ')
    if not os.path.exists(blank_path + '.jpeg'):
        white_screen = (np.ones_like(img_np)*255).astype('uint8')    
        
        print(f'Generating blank canvas...', end=' ')
        for side in range(2):
            if side % 2 == 0:
                n_proj = n_proj_vert
            else:
                n_proj = n_proj_horz

            compiled_tests_blank = bp.compile(white_screen, n_proj, 1, side,
                                                compiled_tests_blank, proj_inplane_offset, proj_outplane_offset,
                                                proj_width, proj_height, trap_path, blank_path, mode)

        disp = cp.asnumpy(compiled_tests_blank)
        disp = (disp*255/disp.max()).astype('uint8')
        np.savez_compressed(blank_path + '.npz', disp=disp)

        image = Image.fromarray(disp)
        image.save(blank_path + '.jpeg')

    blank = np.load(blank_path + '.npz')['disp'].astype('uint8')

    img_np_adjusted = img_np
    img_np_adjusted = (img_np_adjusted / blank)
    img_np_adjusted = (img_np_adjusted*255/img_np_adjusted.max()).astype('uint8')
    img_np_adjusted = cv2.cvtColor(img_np_adjusted, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(img_np_adjusted)
    image.save(img_path_processed)

    print(f'Generating compilation...', end=' ')

    for side in range(4):
        if side % 2 == 0:
            n_proj = n_proj_vert
        else:
            n_proj = n_proj_horz

        compiled_tests = bp.compile(img_np_adjusted, n_proj, downsample_ratio, side,
                                    compiled_tests, proj_inplane_offset, proj_outplane_offset,
                                    proj_width, proj_height, trap_path, path, mode)

    np.savez_compressed(img_path_final  + '.npz', compiled_tests=compiled_tests)
    
    disp = (compiled_tests*255/compiled_tests.max()).astype('uint8')
    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(disp)
    image.save(img_path_final  + '.jpeg')

    return disp