from utils import predbytiles, probav_image_operational
from utils import l8image
from datetime import datetime, timezone
import rasterio
from rasterio import windows
from rasterio import warp
import numpy as np
import h5py
import os
import tensorflow as tf
import logging


BANDS_L8_PV = [1, 2, 4, 5, 6]

SIZE_KERNEL_PSF = (15, 15)
DOWNSAMPLING_PSF = (3, 3)


def model_subsampling_psf_fun(ip, strides=DOWNSAMPLING_PSF, size_kernel=SIZE_KERNEL_PSF):

    number_inputs = tf.keras.backend.int_shape(ip)[-1]

    paddings = tf.constant([[0, 0],
                            [size_kernel[0] // 2, size_kernel[0] // 2],
                            [size_kernel[1] // 2, size_kernel[1] // 2],
                            [0, 0]])

    ip = tf.pad(ip, paddings, mode="REFLECT")

    distances = np.ones(SIZE_KERNEL_PSF)
    pos_centro = tuple(np.array(SIZE_KERNEL_PSF) // 2)

    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            distances[i, j] = np.sqrt((pos_centro[0] - i) ** 2 + (pos_centro[1] - j) ** 2)

    distances *= 30 # Convert distances to meters!
    pesos = np.ndarray(distances.shape + (number_inputs, 1), dtype=tf.keras.backend.floatx())

    for i, sigma in enumerate([46.1236, 50.0845, 46.7560, 80.7639]):
        pesos[:, :, i, 0] = np.exp(-(distances ** 2) / (2 * (sigma ** 2)))
        pesos[:, :, i, 0] /= np.sum(pesos[:, :, i, 0])

    pesos = tf.constant(pesos)

    out = tf.nn.depthwise_conv2d(ip,pesos,strides=(1,)+strides+(1,),
                                 padding="VALID",name="psf")

    return out


class PSFLayer(tf.keras.layers.Layer):
    def __init__(self, strides=DOWNSAMPLING_PSF,size_kernel=SIZE_KERNEL_PSF, **kwargs):
        self.strides = strides
        self.size_kernel = size_kernel
        super(PSFLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return model_subsampling_psf_fun(inputs, strides=self.strides,
                                         size_kernel=self.size_kernel)

    def get_config(self):
        config = super(PSFLayer,self).get_config()
        config["strides"] = self.strides
        config["size_kernel_lanczos"] = self.size_kernel_lanczos
        return config


def model_subsampling_psf():
    input_ = tf.keras.layers.Input((None,None,4))
    out = PSFLayer()(input_)
    return tf.keras.models.Model(inputs=[input_], outputs=[out], name="subsamplingpfs")


def compute_blue_mean(img):
    weights = np.array([0.24916137667938665, 0.7508386233206134])

    # weighted sum Blue bands depending on the spectral response
    img[..., 1] = np.sum(img[..., :2] * weights, axis=-1)
    return img[..., 1:]


RESOLUTION_STEP = {
    "100M": 0.00297619047619/3,
    "333M": 0.00297619047619,
    "1KM": 0.00297619047619*3,
}

CONV_FILTER = np.ones((3, 3), np.float32)
CONV_FILTER[0, 0] = 0.5
CONV_FILTER[0, -1] = 0.5
CONV_FILTER[-1, 0] = 0.5
CONV_FILTER[-1, -1] = 0.5
CONV_FILTER[1, 1] = 1.5

FILTER_CLOUDS = CONV_FILTER.copy()
FILTER_CLOUDS /= np.sum(FILTER_CLOUDS)


def model_labels_hrtolr_fun(ip_, ip_invalid=None, add_padding=False):
    out_invalid = None
    if add_padding:
        paddings = tf.constant([[0, 0],
                                [1, 1],
                                [1, 1],
                                [0, 0]])
    if ip_invalid is not None:
        if add_padding:
            ip_invalid = tf.pad(ip_invalid, paddings, mode="CONSTANT", constant_values=1)

        out_invalid = tf.nn.conv2d(ip_invalid,
                                   CONV_FILTER[..., np.newaxis, np.newaxis],
                                   strides=(1, 3, 3, 1,),
                                   padding="VALID")

    if add_padding:
        ip_ = tf.pad(ip_, paddings, mode="CONSTANT", constant_values=.5)

    out_ = tf.nn.conv2d(ip_,
                        FILTER_CLOUDS[..., np.newaxis, np.newaxis],
                        strides=(1, 3, 3, 1,),
                        padding="VALID")

    if out_invalid is not None:
        return out_, out_invalid

    return out_


def model_labels_hrtolr():
    ip_ = tf.keras.layers.Input((None, None, 1))
    ip_invalid = tf.keras.layers.Input((None, None, 1))

    outs_ = tf.keras.layers.Lambda(lambda x: model_labels_hrtolr_fun(
        x[0], x[1], add_padding=True))([ip_, ip_invalid])
    return tf.keras.models.Model(inputs=[ip_, ip_invalid],
                                 outputs=outs_)


def downscale_mask(ip_, add_padding=False):
    if add_padding:
        paddings = tf.constant([[0, 0],
                                [1, 1],
                                [1, 1],
                                [0, 0]])
        ip_ = tf.pad(ip_, paddings, mode="CONSTANT", constant_values=1)

    return tf.nn.conv2d(ip_,
                        FILTER_CLOUDS[..., np.newaxis, np.newaxis],
                        strides=(1, 3, 3, 1,),
                        padding="VALID")


def model_downscale_mask():
    ip_ = tf.keras.layers.Input((None, None, 1))

    out_ = tf.keras.layers.Lambda(
        lambda x: downscale_mask(x, add_padding=True))(ip_)
    return tf.keras.models.Model(inputs=[ip_, ],
                                 outputs=[out_])


def convert_landsat_to_probav(l8obj, resolution="333M", threshold_invalid=.2, file_out=None):
    """
    Convert Landsat-8 data from Biome or 38-Clouds dataset to same spectral and spatial resolution as Proba-V.
    Following the transformation proposed in:
    Transferring deep learning models for cloud detection between Landsat-8 and Proba-V
    https://www.sciencedirect.com/science/article/abs/pii/S0924271619302801

    :param l8obj: l8 image object.
    :param resolution: one of {333M, 100M, 1KM}
    :param threshold_invalid: Threshold to mask as invalid a pixel
    :return:
    """
    assert (file_out is None) or (not os.path.exists(file_out)
                                  ), "file %s exists will not be overwritten" % file_out
    transform_landsat = l8obj.transform
    transform_landsat = rasterio.Affine(*transform_landsat.flatten().tolist())
    src_crs = l8obj.crs_proj()

    img = l8obj.load_bands(sun_elevation_correction=True,
                           bands=BANDS_L8_PV)

    img = compute_blue_mean(img)
    mask_img2d = np.any(np.ma.getmaskarray(img), axis=-1)
    mask_img2d_float = mask_img2d.astype(np.float32)[..., np.newaxis]

    # Apply CNN to convert from 30m->90m with PSF
    modelo = model_subsampling_psf()
    img = np.ma.filled(img, 0)
    img = modelo.predict(img[np.newaxis, :], batch_size=1)[0]

    # Transform clouds and mask  with simple 3x3 strided mean
    model_for_masks = model_downscale_mask()
    mask_img2d_downsampled_float = model_for_masks.predict(
        mask_img2d_float[np.newaxis], batch_size=1)[0]
    assert mask_img2d_downsampled_float.shape[:2] == img.shape[:2], "Shape img and mask differ {} {}".format(
        img.shape, mask_img2d_downsampled_float.shape)

    # No need to take into account size of filter because it is padded before filtering
    transform_landsat_after_downsampling = rasterio.Affine(transform_landsat.a * 3,
                                                           transform_landsat.b,
                                                           transform_landsat.c,
                                                           transform_landsat.d, transform_landsat.e * 3,
                                                           transform_landsat.f)

    # Resize to mimic PV proyection
    img_resize, transform_probav = _resize_crs_probav(img, src_crs,
                                                      transform_landsat_after_downsampling,
                                                      RESOLUTION_STEP[str(resolution)])

    mask_img2dres, _ = _resize_crs_probav(mask_img2d_downsampled_float, src_crs,
                                          transform_landsat_after_downsampling,
                                          RESOLUTION_STEP[str(resolution)])

    mask_img2dres_bool = (mask_img2dres > threshold_invalid) | (
        mask_img2dres == -1)
    mask_img3dres_bool = predbytiles.mask_2D_to_3D(mask_img2dres_bool[..., 0],
                                                   img_resize.shape[-1])

    img_resize = np.ma.masked_array(img_resize,
                                    (img_resize == -1) | mask_img3dres_bool)

    if file_out is not None:
        write_landsat_as_pv(file_out,
                            img_resize.filled(-1),
                            transform_probav,
                            l8obj)

    if hasattr(l8obj, "load_clouds"):
        clouds = l8obj.load_clouds()
        mask_clouds = np.ma.getmaskarray(clouds) | mask_img2d
        # Convert all masks to float, add last channel
        clouds = clouds.astype(np.float32)[..., np.newaxis]
        mask_clouds = mask_clouds.astype(np.float32)[..., np.newaxis]

        model_for_cloud_masks = model_labels_hrtolr()
        clouds = np.ma.filled(clouds, 0.5)
        clouds, mask_clouds = model_for_cloud_masks.predict(
            [clouds[np.newaxis, :], mask_clouds[np.newaxis, :]], batch_size=1)
        clouds = clouds[0]
        mask_clouds = mask_clouds[0]
        assert clouds.shape[:2] == img.shape[:2], "Shape img and mask clouds {} {}".format(
            img.shape, clouds.shape)
        assert mask_clouds.shape[:2] == img.shape[:2], "Shape img and mask clouds {} {}".format(
            img.shape, mask_clouds.shape)

        clouds_resize, _ = _resize_crs_probav(clouds,
                                              src_crs,
                                              transform_landsat_after_downsampling,
                                              RESOLUTION_STEP[str(resolution)])

        mask_clouds_resize, _ = _resize_crs_probav(mask_clouds,
                                                   src_crs,
                                                   transform_landsat_after_downsampling,
                                                   RESOLUTION_STEP[str(resolution)])

        clouds_resize = np.ma.masked_array(np.clip(clouds_resize, 0, 1),
                                           (clouds_resize == -1) | (mask_clouds_resize > threshold_invalid) | (
            mask_clouds_resize == -1))

        if file_out is not None:
            write_cloud_mask(file_out,
                             clouds_resize[..., 0].filled(-1),
                             transform_probav, l8obj)
    else:
        logging.info("L8 object does not have load clouds attribute")
        clouds_resize = None

    return img_resize, clouds_resize


def _resize_crs_probav(img, src_crs, transform_landsat, resolution_step):
    assert (len(img.shape) ==
            3), "Prepared to work with 3d arrays in HWC format {}".format(img.shape)
    assert not np.any(np.ma.getmask(
        img)), "masked array not supported. fill before calling!"

    window_landsat = (slice(0, img.shape[0]), slice(0, img.shape[1]))
    bbox = windows.bounds(window_landsat,
                          transform_landsat)
    transform, width, height = warp.calculate_default_transform(src_crs,
                                                                {"init": "epsg:4326"},
                                                                img.shape[1], img.shape[0],
                                                                left=bbox[0], bottom=bbox[1],
                                                                right=bbox[2], top=bbox[3],
                                                                resolution=resolution_step)

    img = np.ma.filled(img, fill_value=-1)
    out_array = np.ndarray((img.shape[2], height, width),
                           dtype=img.dtype)

    warp.reproject(np.transpose(img, axes=(2, 0, 1)),
                   out_array, src_crs=src_crs,
                   dst_crs={"init": "epsg:4326"},
                   src_transform=transform_landsat,
                   dst_transform=transform,
                   resampling=warp.Resampling.lanczos,
                   dst_nodata=-1,
                   src_nodata=-1)

    out_array = np.transpose(out_array, (1, 2, 0))

    return out_array, transform


def write_landsat_as_pv(filename, img, transform, piorigen, chunks=(256, 256)):
    assert (len(img.shape) == 3) and (
        img.shape[2] == 4), "Unexpected shape for image {}".format(img.shape)
    assert not np.any(np.ma.getmask(
        img)), "masked array not supported. fill before calling!"

    for i in range(2):
        if img.shape[i] < chunks[i]:
            chunks = None
            break

    with h5py.File(filename, "w") as output:
        output.require_group('crs')
        output['crs'].attrs['GeoTransform'] = np.array(["%f %f %f %f %f %f" % (transform.c,
                                                                               transform.a,
                                                                               transform.b,
                                                                               transform.f,
                                                                               transform.d,
                                                                               transform.e)]).astype('S258')

        output.attrs["image"] = str(piorigen)
        output.attrs["class_image"] = str(piorigen.__class__.__name__)
        output.attrs["END_DATE"] = piorigen.end_date.strftime(
            "%Y-%m-%d %H:%M:%S")
        output.attrs["START_DATE"] = piorigen.start_date.strftime(
            "%Y-%m-%d %H:%M:%S")
        output.attrs["MAP_PROJECTION"] = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
        output.attrs["SLICE"] = np.array([[0, s] for s in img.shape[:2]])

        dsets_copiar = ['LEVEL2A/RADIOMETRY/%s/TOA' %
                        b for b in ["BLUE", "RED", "NIR", "SWIR"]]

        for i, b in enumerate(dsets_copiar):
            output.require_group(os.path.dirname(b))

            dset = output.create_dataset(b,
                                         data=np.round(img[..., i]*2000).astype(np.int16),
                                         chunks=chunks,
                                         compression="gzip")

            dset.attrs["MAPPING"] = np.array([b'Geographic Lat/Lon', b'0.5', b'0.5', str(transform.c),
                                              str(transform.f), str(
                                                  np.abs(transform.a)),
                                              str(np.abs(transform.e)),
                                              b'WGS84', b'Degrees'],
                                             dtype="S")
            dset.attrs["OFFSET"] = 0
            dset.attrs["SCALE"] = 2000


def write_cloud_mask(filename, clouds, transform, chunks=(256, 256)):
    assert len(clouds.shape) == 2, "Unexpected shape for cloud mask {}".format(
        clouds.shape)
    assert not np.any(np.ma.getmask(
        clouds)), "masked array not supported. fill before calling!"
    with h5py.File(filename, "r+") as output:
        dset = output.create_dataset("CM_RESAMPLED",
                                     data=clouds,
                                     chunks=chunks,
                                     compression="gzip")

        dset.attrs["MAPPING"] = np.array([b'Geographic Lat/Lon', b'0.5', b'0.5', str(transform.c),
                                          str(transform.f), str(
                                              np.abs(transform.a)),
                                          str(np.abs(transform.e)),
                                          b'WGS84', b'Degrees'],
                                         dtype="S")


class LandsatAsPVImage(probav_image_operational.ProbaVImageOperational):
    def __init__(self, hdf5_file, soft_mask=False):
        probav_image_operational.ProbaVImageOperational.__init__(self, hdf5_file)
        self.start_date = datetime.strptime(
            self.metadata["START_DATE"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        self.end_date = datetime.strptime(self.metadata["END_DATE"], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc)
        self.map_projection_wkt = self.metadata["MAP_PROJECTION"]
        self.soft_mask = soft_mask
        self.resolution = os.path.basename(os.path.splitext(hdf5_file)[0]).split("_")[-1]
        if self.resolution not in ["100M", "333M", "1KM"]:
            logging.warning("Could not interpret resolution string %s" % self.resolution)
            self.resolution = None

    def load_clouds(self, threshold=.5, slice_=None):
        if slice_ is None:
            slice_ = (slice(None), slice(None))

        with h5py.File(self.hdf5_file, "r") as input_f:
            clouds = input_f["CM_RESAMPLED"][slice_]

        clouds = np.ma.masked_array(np.clip(clouds, 0, 1), np.abs(clouds - -1) < 1e-5)

        if not self.soft_mask:
            clouds = (clouds > threshold).astype(np.float32)

        return clouds

    def Landsat8ACCA(self):
        path_2_file = self.metadata["image"]
        class_name = self.metadata["class_image"]

        if class_name == "Biome":
            return l8image.Biome(path_2_file)
        elif class_name == "L8_38Clouds":
            return l8image.L8_38Clouds(path_2_file)

        raise FileNotFoundError(
            "Could not find Landsat files for object %s" % self.name)

    def load_bands(self, bands=None, with_mask=True, slice_=None):
        imgr = probav_image_operational.ProbaVImageOperational.load_bands(self, bands=bands,
                                                                          with_mask=False,
                                                                          slice_=slice_)
        if with_mask:
            return np.ma.masked_array(imgr, imgr <= -.5)
        return imgr

    def load_mask(self, slice_=None):
        if slice_ is None:
            slice_ = (slice(None), slice(None))

        with h5py.File(self.hdf5_file, "r") as input_f:
            clouds = input_f["CM_RESAMPLED"][slice_]

        return np.abs(clouds - -1) < 1e-5