import numpy as np
import itertools
from numpy.lib.stride_tricks import as_strided


def mask_2D_to_3D(mascara, nchannels):
    return as_strided(mascara,
                      mascara.shape + (nchannels,),
                      mascara.strides + (0,))


def predict(pred_function, pv_image, tilesize=1000, buffersize=0):
    shape = (pv_image.nrows, pv_image.ncols)

    mask_save_valid = np.zeros(shape, dtype=np.bool)
    output_save = None
    # predict in tiles of tile_size x tile_size pixels
    for i, j in itertools.product(range(0, shape[0], tilesize), range(0, shape[1], tilesize)):
        slice_current = (slice(i, min(i + tilesize, shape[0])),
                         slice(j, min(j + tilesize, shape[1])))
        slice_pad = (slice(max(i - buffersize, 0), min(i + tilesize + buffersize, shape[0])),
                     slice(max(j - buffersize, 0), min(j + tilesize + buffersize, shape[1])))

        slice_save_i = slice(slice_current[0].start - slice_pad[0].start,
                             None if (slice_current[0].stop - slice_pad[0].stop) == 0 else slice_current[0].stop -
                                                                                           slice_pad[0].stop)
        slice_save_j = slice(slice_current[1].start - slice_pad[1].start,
                             None if (slice_current[1].stop - slice_pad[1].stop) == 0 else slice_current[1].stop -
                                                                                           slice_pad[1].stop)

        # slice_save is normally slice(buffersize,-buffersize),slice(buffersize,-buffersize) except in the borders
        slice_save = (slice_save_i, slice_save_j)

        image_probav_with_mask = pv_image.load_bands(slice_=slice_pad)

        mascaravalid = ~np.any(np.ma.getmaskarray(image_probav_with_mask),
                               axis=-1, keepdims=True)

        mascaravalidcurrent = mascaravalid[slice_save]

        # call predict only if there are pixels not invalid
        if np.any(mascaravalidcurrent):
            mask_save_valid[slice_current] = mascaravalidcurrent
            vals_to_predict = np.ma.filled(image_probav_with_mask, 0)

            pred_continuous_tf = pred_function([vals_to_predict, mascaravalid])[slice_save]

            if output_save is None:
                output_save = np.zeros(shape+(pred_continuous_tf.shape[-1],), dtype=pred_continuous_tf.dtype)

            output_save[slice_current] = pred_continuous_tf

    return np.ma.masked_array(output_save,
                              mask_2D_to_3D(~mask_save_valid, output_save.shape[-1]))
