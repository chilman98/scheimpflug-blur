import numpy as np
import cupy as cp
import cv2
import os
import time
from PIL import Image
from scipy.ndimage import rotate
import generateSkewedPixels as sp
import generateColumns as col

def compile(img_np, n_projections, downsample_ratio, side, compiled_tests,
                           proj_inplane_offset, proj_outplane_offset, proj_width, proj_height, trap_path, path, mode):
    
    blank = np.average(img_np) == img_np.max()
    rotation = side*90
    img_np_rot = rotate(img_np, angle=rotation, axes=(0, 1))
    
    image_height = img_np_rot.shape[1]
    image_width = img_np_rot.shape[0]

    proj_offaxis_offset_list = range(0, image_width+1,image_width//(n_projections+1))[1:-1]
    print(' ')
    print(f'Initializing compilation for side {side+1}...')
    skewed_pixels = sp.generate(proj_width+1, proj_height+1,
                                         proj_inplane_offset, proj_outplane_offset,
                                        image_height, plot=False)
    skewed_pixels_ds = sp.generate(int(proj_width/downsample_ratio+1),
                                            int(proj_height/downsample_ratio+1),
                                            proj_inplane_offset, proj_outplane_offset,
                                            image_height, plot=False)
    skewed_pixels_scaled = (skewed_pixels*image_height/skewed_pixels[:,:,1].max()).astype('uint32')

    trap_path = trap_path + f'/image({image_width}, {image_height})/proj({proj_width}, {proj_height})'
    trap_path = trap_path + f'/xy-offset({round(proj_inplane_offset,2)}, {round(proj_outplane_offset,2)})'
    path = path + f'/xy-offset({round(proj_inplane_offset,2)}, {round(proj_outplane_offset,2)})'
    rot_path = path + f'/snips_rotation{rotation}'
    column_path = rot_path + f'/snipcolumns'
    os.makedirs(column_path, exist_ok=True)

    trap_origins = [[square[:, 0].min(), square[:, 1].min()] for square in skewed_pixels_scaled]
    trap_columns = [origin[1] for origin in trap_origins]
    trap_columns = cp.array(trap_columns)
    trap_columns = cp.append(trap_columns.reshape((proj_height, proj_width))[:,0], image_height)

    test_height = np.ceil(skewed_pixels_scaled[:,:,1].max()/2)*2
    test_width = np.ceil(skewed_pixels_scaled[:,:,0].max()/2)*2
    test_shape = (int(test_width), int(test_height))
    
    t1 = time.time()
    print(' ')

    disp = generateEdge(trap_columns, skewed_pixels, skewed_pixels_scaled, skewed_pixels_ds, trap_origins,
                                  img_np_rot, image_height, image_width, proj_height, proj_width,
                                  test_shape, proj_inplane_offset, proj_outplane_offset, proj_offaxis_offset_list, n_projections,
                                  downsample_ratio, rot_path, column_path, trap_path, path, mode)
    
    disp = rotate(cp.asnumpy(disp), angle=-side*90, axes=(0, 1))

    if len(compiled_tests) == 0:
        compiled_tests = np.zeros((disp.shape[0], disp.shape[1], 3), dtype='uint64')
    
    compiled_tests += disp
    if blank:
        compiled_tests += rotate(cp.asnumpy(disp), angle=180, axes=(0, 1))
    print(' ')
        
    t2 = time.time()
    print(f'Execution time for edge compilation: {round((t2 - t1)/60, 3)} min')
    print(' ')

    return compiled_tests

def generateEdge(trap_columns, skewed_pixels, skewed_pixels_scaled, skewed_pixels_downsample, trap_origins,
                           img_np, image_height, image_width, proj_height, proj_width, test_shape, proj_inplane_offset, proj_outplane_offset,
                           proj_offaxis_offset_list, n_projections, downsample_ratio,
                           rot_path, column_path, trap_path, path, mode):
    
    os.makedirs(f'{rot_path}_sim', exist_ok=True)
    os.makedirs(f'{rot_path}_proj', exist_ok=True)

    blank = np.average(img_np) == img_np.max()
    img_cp = cp.array(img_np)

    compiled_tests_edge = cp.zeros((image_width, image_height, 3), dtype='uint64')
    for offset_number, proj_offaxis_offset in enumerate(proj_offaxis_offset_list):
        if blank:
            disp_path = f'{rot_path}_downsample{downsample_ratio}'
        else:
            disp_path = f'{rot_path}_sim/downsample{downsample_ratio}_{mode}_simoffset{proj_offaxis_offset}'
            
        if not os.path.exists(disp_path + '.npz'):
            img_extd = cp.zeros((test_shape[0] + img_cp.shape[0], test_shape[1], 3), dtype='uint8')
            img_extd[test_shape[0]//2 : test_shape[0]//2 + img_cp.shape[0], :, :] = img_cp
        
    for offset_number, proj_offaxis_offset in enumerate(proj_offaxis_offset_list):
        img_extd = cp.zeros((test_shape[0] + img_cp.shape[0], test_shape[1], 3), dtype='uint8')
        img_extd[test_shape[0]//2 : test_shape[0]//2 + img_cp.shape[0], :, :] = img_cp

        if blank:
            disp_path = f'{rot_path}_downsample{downsample_ratio}'
        else:
            disp_path = f'{rot_path}_sim/downsample{downsample_ratio}_{mode}_simoffset{proj_offaxis_offset}'

        if os.path.exists(disp_path + '.npz'):
            print(' ')
            print(f'Loading projection {offset_number+1}/{n_projections}...', end=' ')
        else:
            trap_column_path = trap_path + '/trapcolumns1'

            offset_path = rot_path + f'/z-offset({proj_offaxis_offset})'
            canvas_path = rot_path + f'/canvas({proj_offaxis_offset})_{mode}.npz'
            
            if blank:
                proj_canvas = (cp.ones((proj_width, proj_height, 3))*255).astype('uint8')
                img_extd = (cp.ones_like(img_extd)*255).astype('uint8')
            else:
                proj_canvas = generateBackProjection(trap_columns, skewed_pixels, cp.array(trap_origins),
                                                    img_extd, column_path, image_height, image_width, proj_height,
                                                    proj_width, offset_path, proj_offaxis_offset, offset_number,
                                                    n_projections, test_shape, canvas_path, trap_column_path, mode)

            proj_disp = cp.asnumpy(proj_canvas)
            proj_disp = cv2.cvtColor(proj_disp, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(proj_disp)
            image = image.rotate(-90, expand=True)
            image.save(f'{rot_path}_proj/downsample{downsample_ratio}_{mode}_projoffset{proj_offaxis_offset}.jpeg')

            skewed_pixels_scaled = cp.reshape(skewed_pixels_scaled, (proj_height, proj_width, 4, 2))
            square = [skewed_pixels_scaled[ 0, 0, 0],
                      skewed_pixels_scaled[ 0,-1, 1],
                      skewed_pixels_scaled[-1,-1, 1],
                      skewed_pixels_scaled[-1, 0, 1]]

            trapezoid = cp.array(col.drawTrapezoid(square, 10)).T
            trap_mask = cp.zeros_like(img_extd)
            if blank: upper_bound = image_width//2
            else: upper_bound = proj_offaxis_offset
            lower_bound = upper_bound + trapezoid.shape[0]

            trap_mask[upper_bound:lower_bound, :trapezoid.shape[1]] = cp.stack([trapezoid]*3, axis=2)
            img_extd_masked = img_extd
            img_extd_masked[trap_mask == 0] = trap_mask[trap_mask == 0]

            trap_column_path = trap_path + f'/trapcolumns{downsample_ratio}'

            t1 = time.time()
            disp = convertProjToDisp(img_extd_masked, proj_canvas, proj_inplane_offset, proj_outplane_offset, proj_offaxis_offset, skewed_pixels_downsample, image_height,
                                    image_width, downsample_ratio, test_shape, trap_column_path, mode, blank)
            
            if not blank:
                disp = disp[(disp.shape[0]-image_width)//2:(disp.shape[0]+image_width)//2]

            t2 = time.time()
            print(' ')
            print(f'Execution time for conversion: {round(t2 - t1, 3)} sec')
            
            disp = cp.asnumpy(disp)
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            
            np.savez_compressed(disp_path + '.npz', disp=disp)

            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(disp)
            image.save(disp_path + '.jpeg')

        disp = cp.array(np.load(disp_path + '.npz')['disp'], dtype='uint8')

        if blank:
            left_bound = int(disp.shape[0]/2 + proj_offaxis_offset - image_width)
            compiled_tests_edge += cp.array(disp)[left_bound:left_bound + image_width]
        else:
            compiled_tests_edge += cp.array(disp)

    disp = compiled_tests_edge.get()
    # disp = (disp*255/disp.max()).astype('uint64')    

    return disp

def generateBackProjection(trap_columns, skewed_pixels, trap_origins, img_extd,
                            column_path, image_height, image_width, proj_height, proj_width, offset_path,
                            proj_offaxis_offset, offset_number, n_projections, test_shape, canvas_path, trap_path, mode):
    
    if os.path.exists(canvas_path):
        print(' ')
        print(f'Loading saved projection {offset_number+1}/{n_projections}...', end=' ')
    else:
        ta = time.time()     
        print(' ')
        print(f'Generating projection {offset_number+1}/{n_projections}...', end=' ')
        
        t1 = time.time()
        n_layers = proj_width
        count_int = (len(trap_columns)-1)//10
        trap_rows = trap_origins[:,0].reshape(-1, n_layers).T
        proj_canvas = cp.zeros((proj_width, proj_height, 3), dtype='uint8')

        for n, column in enumerate(trap_columns[:-1]):
            if (n+1) % count_int == 0 or n == len(trap_columns)-2:
                print(' ')
                print(f'Backprojecting column {n+1}/{len(trap_columns)-1}...', end=' ')
                
            proj_column, n = col.process(n, column, trap_columns, trap_rows, skewed_pixels, trap_origins,
                                            n_layers, img_extd[:,trap_columns[n]:trap_columns[n+1]+1,:],
                                            image_height, image_width, column_path, offset_path, trap_path,
                                            proj_offaxis_offset, test_shape, mode)
            proj_canvas[:, n, :] = proj_column

            if (n+1) % count_int == 0:
                t2 = time.time()
                average_cost = round((t2 - t1)*1000/count_int)
                remaining_cost = len(trap_columns)-n-1
                print(f'average cost: {average_cost} ms', end=', ')
                print(f'est time remaining: {round(average_cost*remaining_cost/1000)} s', end=' ')
                t1 = time.time()
                
        tb = time.time()
        print(' ')
        print(f'Execution time for backprojection: {round(tb - ta, 3)} sec', end=' ')

        np.savez_compressed(canvas_path, proj_canvas=proj_canvas)

    proj_canvas = cp.array(np.load(canvas_path)[f'proj_canvas'], dtype='uint8')

    return proj_canvas

def convertProjToDisp(img_extd, proj_canvas, proj_inplane_offset, proj_outplane_offset, proj_offaxis_offset, skewed_pixels_downsample, image_height,
                      image_width, downsample_ratio, test_shape, trap_path, mode, blank):
    
    proj_width = test_shape[0]
    compiled_columns = cp.zeros((2*max(proj_width, image_width), test_shape[1], 3), dtype='uint64')

    proj_canvas = downsampleProjection(proj_canvas, downsample_ratio, mode)
    skewed_pixels_scaled = (skewed_pixels_downsample*image_height/skewed_pixels_downsample[:,:,1].max()).astype('uint16')
    trap_origins = [[square[:, 0].min(), square[:, 1].min()] for square in skewed_pixels_scaled]
    trap_columns = [origin[1] for origin in trap_origins]
    trap_columns = cp.array(trap_columns).reshape((proj_canvas.shape[1], proj_canvas.shape[0]))[:,0]
    n_layers = proj_canvas.shape[0]
    count_int = (len(trap_columns)-1)//10
    t1 = time.time()
    if blank: proj_offaxis_offset = image_width//2

    print(' ')
    for n, column in enumerate(trap_columns):
        if (n+1) % (count_int+1) == 0 or n == len(trap_columns)-1:
            print(' ')
            print(f'Converting column {n+1}/{len(trap_columns)}...', end=' ')
        
        trap_column = col.generateTrapezoidColumn(skewed_pixels_downsample, image_height, n_layers, (column, n), trap_path)
        test_column = cp.zeros((proj_width, trap_column.shape[1], 3, n_layers - 1), dtype='float16')
        trap_column = cp.stack([trap_column]*3, axis=3)
        proj_column = cp.array(proj_canvas[:, n, :])[cp.newaxis, cp.newaxis, :]

        trap_column = cp.multiply(trap_column/(255), proj_column)
        origin_offset = trap_origins[n*n_layers:(n+1)*n_layers-1]

        column_ranges = [slice(offset[0], offset[0] + trap_column.shape[0]) for offset in origin_offset]
        for pixel, column_range in enumerate(column_ranges):
            test_column[column_range, :, :, pixel] = trap_column[:, :, pixel]
            
            # print(f'{pixel}')
            # test_column2 = cp.max(test_column, axis=3)
            # plt.imshow(test_column2[column_range, :, :].get())
            # plt.axis('off')
            # plt.show()

        test_column = cp.max(test_column, axis=3)

        upper_bound = abs(int(proj_width)-image_width)//2 + proj_offaxis_offset
        lower_bound = upper_bound + test_column.shape[0]
        left_bound = trap_columns[n]
        right_bound = left_bound + test_column.shape[1]

        column_sum = cp.sum(compiled_columns[upper_bound:lower_bound, right_bound-1:right_bound, 0])
        test_sum = test_column[:,-2:-1]
        test_sum = cp.sum(test_sum)

        if column_sum > test_sum:
            upper_bound = cp.abs(compiled_columns.shape[0] - img_extd.shape[0])//2
            lower_bound = upper_bound + img_extd.shape[0]
            compiled_columns[upper_bound:lower_bound, left_bound:] = img_extd[:img_extd.shape[0], left_bound:]
            break
        else:
            compiled_columns[upper_bound:lower_bound, left_bound:right_bound] = test_column

        if (n+1) % (count_int+1) == 0:
            
            t2 = time.time()
            average_cost = round((t2 - t1)*1000/count_int)
            remaining_cost = len(trap_columns)-n+1
            print(f'average cost: {average_cost} ms', end=', ')
            print(f'est time remaining: {round(average_cost*remaining_cost/1000)} s', end=' ')
            t1 = time.time()

    x0, y0, z0 = image_height + proj_inplane_offset, proj_outplane_offset, proj_offaxis_offset + (compiled_columns.shape[0] - image_width)//2
    z, x = np.meshgrid(np.arange(compiled_columns.shape[0]), np.arange(compiled_columns.shape[1]), indexing='ij')
    dist = 1/np.sqrt((x - x0)**2 + (z - z0)**2 + y0**2)

    compiled_columns = compiled_columns*cp.stack([cp.asarray(dist)]*3, axis=2)
    compiled_columns = compiled_columns/compiled_columns.max()

    disp = compiled_columns
    disp = (disp*255/disp.max()).astype('uint8')

    return disp

def downsampleProjection(proj_img, downsample_ratio, mode):
    proj_height, proj_width = (proj_img.shape[1], proj_img.shape[0])
    new_height, new_width = (proj_height//downsample_ratio,
                             proj_width//downsample_ratio)
    downsampled_img = cp.zeros((new_width, new_height, 3))

    reshaped_img = proj_img.reshape(new_width, downsample_ratio, new_height, downsample_ratio, 3)
    reshaped_img = reshaped_img.transpose(0, 2, 1, 3, 4)
    downsampled_img = cp.nanmean(reshaped_img, axis=(2, 3))

    if mode == 'min':
        downsampled_img = cp.nanmin(reshaped_img, axis=(2, 3))
    elif mode == 'max':
        downsampled_img = cp.nanmax(reshaped_img, axis=(2, 3))
    elif mode == 'ave':
        downsampled_img = cp.nanmean(reshaped_img, axis=(2, 3))


    return downsampled_img