import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import generateForwardProjections as fp

def main(img_path):
    global downsample_ratios_list, inplane_offset_ratio, mode
    global outplane_offset_ratio, proj_width, proj_height, n_projections_list

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img_np = cv2.imread(img_path)
    aspect_ratio = img_np.shape[0]/img_np.shape[1]
    proj_inplane_offset = img_np.shape[1]*inplane_offset_ratio
    proj_outplane_offset = img_np.shape[1]*outplane_offset_ratio

    for n_projections in n_projections_list:
        trap_path = f'renderdata/trapezoidcolumns'
        blank_path = f'renderdata/blanks/{img_np.shape}_blank/{n_projections}_xy-offset({round(proj_inplane_offset,2)}, {round(proj_outplane_offset,2)})'
        path = f'renderdata/{img_name}/proj({proj_width}, {proj_height})_{n_projections}'
        os.makedirs(path, exist_ok=True)

        n_proj_vert = round(n_projections*aspect_ratio/(2*aspect_ratio+2))
        n_proj_horz = n_projections//2 - n_proj_vert

        for downsample_ratio in downsample_ratios_list:
            img_path_processed = f'{path}_processed_xy-offset({round(proj_inplane_offset,2)}, {round(proj_outplane_offset,2)}).jpeg'
            img_path_final = f'{path}_downsample{downsample_ratio}_{mode}_compiled_xy-offset({round(proj_inplane_offset,2)}, {round(proj_outplane_offset,2)})'

            if os.path.exists(img_path_final + 'jpeg'): continue

            t1 = time.time()
            fp.compile(img_np, blank_path, n_proj_vert, n_proj_horz, downsample_ratio,
                                proj_inplane_offset, proj_outplane_offset, proj_width,
                                proj_height, trap_path, path, mode, img_path_processed, img_path_final)
            t2 = time.time()
            print(f'Total execution time: {round((t2 - t1)/60, 3)} min')
            print(' ')

            compiled_tests = np.load(img_path_final + '.npz')['compiled_tests'].astype('int64')
            # compiled_tests[:, :, [0, 2]] = compiled_tests[:, :, [2, 0]]

            # disp = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            # disp = (disp*compiled_tests.max()/disp.max()).astype('int64')

            # disp = disp - compiled_tests
            # disp = (disp/disp.max()).astype('float64')

            disp = (compiled_tests*255/compiled_tests.max()).astype('uint8')
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(disp)

            image = Image.fromarray(disp)
            image.save(f'output/{img_name}.jpeg')      
            # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            # axs[0].imshow(disp[:,:,0], interpolation='none', cmap='RdBu')
            # axs[1].imshow(disp[:,:,1], interpolation='none', cmap='RdBu')
            # axs[2].imshow(disp[:,:,2], interpolation='none', cmap='RdBu')
            # plt.tight_layout()
            # plt.show()

def get_files():
    list=[]
    cwd = os.getcwd()
    folder_path = os.path.join(cwd, 'images')
    for filepath,dirnames,filenames in os.walk(folder_path):
        for filename in filenames:
            list.append(os.path.join(filepath,filename))
    return list

if __name__ == '__main__':
    downsample_ratios_list = [1]
    n_projections_list = [4]
    inplane_offset_ratio = 0.1/3
    outplane_offset_ratio = 1
    proj_height = 720//2
    proj_width = 1280//2
    mode = 'ave'

    img_paths=get_files()
    for img_path in img_paths:
        main(img_path)
