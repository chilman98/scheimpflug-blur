import gc
import numpy as np
import cupy as cp
import os
from skimage.draw import polygon

def process(n, column, trap_columns, trap_rows, skewed_pixels, trap_origins, n_layers, img_column,
                  image_height, image_width, column_path, offset_path, trap_path, proj_offaxis_offset, test_shape, mode):

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    
    trap_column = generateTrapezoidColumn(skewed_pixels, image_height, n_layers, (column, n), trap_path, True)

    if trap_column.shape[0] == 0 or trap_column.shape[1] == 0:
        img_column_double = cp.repeat(cp.repeat(img_column, 2, axis=0), 2, axis=1)
        proj_column, n = process(n, column*2, trap_columns*2, trap_rows*2, skewed_pixels,
                                        trap_origins*2, n_layers, img_column_double,
                                        image_height*2, image_width*2, column_path + '/2x', offset_path + '/2x', trap_path + '/2x',
                                        proj_offaxis_offset*2, (test_shape[0]*2, test_shape[1]*2), mode)

    else:    
        os.makedirs(column_path, exist_ok=True)

        snip_column = generateSnippetColumn(img_column, proj_offaxis_offset, trap_rows,
                                            trap_column.shape, test_shape, image_width, (column, n), column_path)
        
        start, stop = (n*n_layers, min((n+1)*n_layers, skewed_pixels.shape[0]))
        proj_column, n = generateBackprojectionColumn(snip_column, trap_column, (start, stop, n), mode)
                
    return proj_column, n

def drawTrapezoid(points, scale_factor = 1, plot = False):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    height = np.ceil((max_y - min_y + 1)).astype('uint16')
    width = np.ceil((max_x - min_x + 1)).astype('uint16')

    canvas = np.zeros((height, width), dtype=np.float32)

    adjusted_points = [(x - min_x, y - min_y) for x, y in points]
    x_coords = [p[0] for p in adjusted_points]
    y_coords = [p[1] for p in adjusted_points]

    rr, cc = polygon(y_coords, x_coords, canvas.shape)
    canvas[rr, cc] = 1.0

    return canvas

def generateTrapezoidColumn(skewed_pixels, image_height, n_layers, proj_column, path, post=False):
    column, n = proj_column
    os.makedirs(path, exist_ok=True)
    column_path = path + f'/trapezoid({column}).npz'

    # t1 = time.time()
    if os.path.exists(column_path):
        trap_column = cp.array(np.load(column_path)['trap_column'], dtype='uint8')
    else:
        skewed_pixels_scaled = (skewed_pixels*image_height/skewed_pixels[:,:,1].max()).astype('uint16')
        skewed_column = skewed_pixels_scaled[n*n_layers:(n+1)*n_layers-1,:,:]
        trap_height, trap_width = (skewed_column[0,:,1].max() - skewed_column[0,:,1].min(),
                                   skewed_column[0,:,0].max() - skewed_column[0,:,0].min()+1)
        trap_column = cp.zeros((trap_width, trap_height, n_layers), dtype='uint8')
        
        for pixel, square in enumerate(skewed_column):
            square = [square[0], square[1],
                     [square[2][0], (square[2][1]+1)],
                     [square[3][0], (square[3][1]+1)]]
            trap_mask = cp.array(drawTrapezoid(square)).T
            trap_mask = (trap_mask[:,:-1]*255).astype(cp.uint8)
            max_width = trap_width if trap_width <= trap_mask.shape[0] else trap_mask.shape[0]
            max_height = trap_height if trap_height <= trap_mask.shape[1] else trap_mask.shape[1]
            trap_column[:max_width, :max_height, pixel] = trap_mask[:max_width, :max_height]
        
        np.savez_compressed(column_path, trap_column=trap_column)
        # print(f'1: {round(tb-ta,3)}, 2: {round(tc-tb,3)}, 3: {round(td-tc,3)}, 4: {round(te-td,3)}, 5: {round(tf-te,3)}')

    # t2 = time.time()
    
    # if n % 2 == 1:
    # if post:
    #     if n % 10 == 9:
    #         print(f'Trapezoid generation: {round((t2 - t1)*100000)} µs,', end='   ')

    return trap_column

def generateSnippetColumn(img_column, proj_offaxis_offset, trap_rows, column_shape, test_shape, image_width, proj_column, path):
    column, n = proj_column
    column_path = path + f'/snippet({column}).npz'

    if os.path.exists(column_path):
        snip_column = cp.array(np.load(column_path)['snip_column'], dtype='uint8')
    else:
        if column_shape[0] == 0:
            return cp.stack([cp.nan(column_shape)]*3, axis=3)
        
        img_float = img_column.astype('float32')
        img_float[:test_shape[0]/2, :, :] = cp.nan
        img_float[test_shape[0]/2 + image_width:, :, :] = cp.nan


        origins = cp.arange(img_column.shape[0])
        origins = cp.stack([origins, cp.full_like(origins, column)], axis=1)
        origins = cp.stack([origins]*column_shape[0], axis=2)
        origins = cp.stack([origins]*column_shape[1], axis=3)

        row_indices = cp.stack([cp.arange(column_shape[0])]*img_float.shape[0], axis=1).T + origins[:,0,:,0]
        col_indices = cp.stack([cp.arange(column_shape[1])]*img_float.shape[0], axis=1).T + origins[:,1,0,:]
        snip_column = img_float[row_indices[:, :, None], col_indices[:, None, :], :].transpose((1, 2, 0, 3))
        np.savez_compressed(column_path, snip_column=snip_column)

    snip_column = snip_column[:,:,trap_rows[:,n] + proj_offaxis_offset,:]

    test_disp = snip_column

    # t2 = time.time()
    # if n % 10 == 9:
    #     # print(f'Column: {column}', end=' ')
    #     print(f'Snippet generation: {round((t2 - t1)*100000)} µs,', end='   ')

    return snip_column

def generateBackprojectionColumn(snip_column, trap_column, pixel_loc, mode):
    trap_column = cp.stack([trap_column]*3, axis=3)
    n = pixel_loc[2]

    if trap_column.shape != snip_column.shape:
        max_width = min(snip_column.shape[0], trap_column.shape[0])
        max_height = min(snip_column.shape[1], trap_column.shape[1])
        snip_column = snip_column[:max_width, :max_height, :]
        trap_column = trap_column[:max_width, :max_height, :]

    snip_column = cp.where(trap_column <= 1, cp.nan, snip_column)

    if mode == 'min':
        slice_colors = cp.nanmin(snip_column, axis=(0, 1))
    elif mode == 'max':
        slice_colors = cp.nanmax(snip_column, axis=(0, 1))
    elif mode == 'ave':
        slice_colors = cp.nanmean(snip_column, axis=(0, 1))

    slice_colors = cp.where(slice_colors == cp.nan, 0, slice_colors)
    proj_column = slice_colors[cp.newaxis, cp.newaxis, :]

    return proj_column, n