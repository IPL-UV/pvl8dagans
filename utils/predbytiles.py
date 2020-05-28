import numpy as np
import itertools
from numpy.lib.stride_tricks import as_strided


def mask_2D_to_3D(mascara, nchannels):
    return as_strided(mascara,
                      mascara.shape + (nchannels,),
                      mascara.strides + (0,))


def predict(pred_function, pv_image, tilesize=1024, buffersize=16):
    shape = (pv_image.nrows, pv_image.ncols)

    mask_save_invalid = np.zeros(shape, dtype=np.bool)
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

        maskcarainvalid = np.any(np.ma.getmaskarray(image_probav_with_mask),
                                 axis=-1, keepdims=False)

        mascarainvalidcurrent = maskcarainvalid[slice_save]

        # call predict only if there are pixels not invalid
        if not np.all(mascarainvalidcurrent):
            mask_save_invalid[slice_current] = mascarainvalidcurrent
            vals_to_predict = np.ma.filled(image_probav_with_mask, 0)

            pred_continuous_tf = pred_function(vals_to_predict)[slice_save]

            if output_save is None:
                output_save = np.zeros(shape+(pred_continuous_tf.shape[-1],), dtype=pred_continuous_tf.dtype)

            output_save[slice_current] = pred_continuous_tf

    return np.ma.masked_array(output_save,
                              mask_2D_to_3D(mask_save_invalid, output_save.shape[-1]))


def padded_predict(model_clouds, module_shape=4):
    """
    Force predict with images multiple of 4 to avoid U-Net concat problems

    :param model_clouds: keras model to predict (small U-Net)
    :param module_shape: image sizes must be multiple of this number
    :return:
    """
    def predict(x):
        shape_image = np.array(list(x.shape))[:2].astype(np.int64)
        shape_padded_image = np.ceil(shape_image.astype(np.float32) / module_shape).astype(np.int64) * module_shape

        if np.all(shape_image == shape_padded_image):
            return model_clouds.predict(x[np.newaxis], batch_size=1)[0, ...]

        pad_to_add = shape_padded_image - shape_image
        x_padded = np.pad(x, [[0, pad_to_add[0]],
                              [0, pad_to_add[1]],
                              [0, 0]], mode="reflect")
        pred_padded = model_clouds.predict(x_padded[np.newaxis], batch_size=1)[0, ...]
        slice_ = (slice(0, shape_padded_image[0]-pad_to_add[0]),
                  slice(0, shape_padded_image[1]-pad_to_add[1]),
                  slice(None))

        return pred_padded[slice_]

    return predict
