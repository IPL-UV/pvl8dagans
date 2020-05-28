import numpy as np
import h5py
from datetime import datetime
import os
import re
from datetime import timezone
from utils import predbytiles


def read_band_toa(input_f, band, slice_to_read):
    attrs = input_f[band].attrs
    if ("OFFSET" in attrs) and ("SCALE" in attrs):
        return (input_f[band][slice_to_read]-attrs["OFFSET"])/attrs["SCALE"]
    return input_f[band][slice_to_read]


class ProbaVImageOperational:
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        try:
            with h5py.File(self.hdf5_file, "r") as input_f:
                # reference metadata: http://www.vito-eodata.be/PDF/image/PROBAV-Products_User_Manual.pdf
                valores_blue = input_f["LEVEL2A/RADIOMETRY/BLUE/TOA"].attrs["MAPPING"][3:7].astype(np.float64)
                self.transform = np.array([[valores_blue[2], 0, valores_blue[0]],
                                           [0, -valores_blue[3], valores_blue[1]]])
                self.shape = input_f["LEVEL2A/RADIOMETRY/BLUE/TOA"].shape
                self.metadata = dict(input_f.attrs)
        except OSError as e:
            raise FileNotFoundError("Error opening file %s" % self.hdf5_file)

        # Same names to keep consistency
        self.ncols = self.shape[1]
        self.nrows = self.shape[0]

        self.name = os.path.basename(self.hdf5_file)

        matches = re.match("PROBAV_L2A_\d{8}_\d{6}_(\d)_(\d..?M)_(V\d0\d)", self.name)

        if matches is not None:
            self.camera, self.resolution, self.version = matches.groups()

        if "OBSERVATION_END_DATE" in self.metadata:
            self.end_date = datetime.strptime(" ".join(self.metadata["OBSERVATION_END_DATE"].astype(str).tolist()+
                                                       self.metadata["OBSERVATION_END_TIME"].astype(str).tolist()),
                                              "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            self.start_date = datetime.strptime(" ".join(self.metadata["OBSERVATION_START_DATE"].astype(str).tolist()+
                                                         self.metadata["OBSERVATION_START_TIME"].astype(str).tolist()),
                                                "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            self.map_projection_wkt = " ".join(self.metadata["MAP_PROJECTION_WKT"].astype(str).tolist())

    def load_bands(self, bands=None, with_mask=True, slice_=None):
        if bands is None:
            bands = (0, 1, 2, 3)

        if slice_ is None:
            slice_ = (slice(None), slice(None))

        bn = ["BLUE", "RED", "NIR", "SWIR"]
        bands_names = ["LEVEL2A/RADIOMETRY/%s/TOA" % bn[i] for i in bands]
        with h5py.File(self.hdf5_file, "r") as input_f:
            bands_arrs = [read_band_toa(input_f, band, slice_) for band in bands_names]
        img = np.stack(bands_arrs, axis=-1)

        if with_mask:
            sm = self.load_sm(slice_)
            mascara = mask_operational(img, sm)
            img = np.ma.masked_array(img,
                                     predbytiles.mask_2D_to_3D(mascara,
                                                               img.shape[-1]))

        return img

    def save_bands(self, img):
        assert img.shape[-1] == 4, "Unexpected shape {}".format(img.shape)

        with h5py.File(self.hdf5_file, "r+") as input_f:
            for i, b in enumerate(["BLUE", "RED", "NIR", "SWIR"]):
                band_to_save = img[..., i]
                mask_band_2_save = np.ma.getmaskarray(img[..., i])
                band_to_save = np.clip(np.ma.filled(band_to_save, 0), 0, 2)
                band_name = "LEVEL2A/RADIOMETRY/%s/TOA" % b
                attrs = input_f[band_name].attrs
                band_to_save *= attrs["SCALE"]
                band_to_save += attrs["OFFSET"]
                band_to_save = np.round(band_to_save).astype(np.int16)
                band_to_save[mask_band_2_save] = -1
                input_f[band_name][...] = band_to_save

    def load_sm(self, slice_=None):
        """
        ## Reference of values in `SM` flags.

        From user manual http://www.vito-eodata.be/PDF/image/PROBAV-Products_User_Manual.pdf pag 67
        * Clear  ->    000
        * Shadow ->    001
        * Undefined -> 010
        * Cloud  ->    011
        * Ice    ->    100
        * `2**3` sea/land
        * `2**4` quality swir (0 bad 1 good)
        * `2**5` quality nir
        * `2**6` quality red
        * `2**7` quality blue
        * `2**8` coverage swir (0 no 1 yes)
        * `2**9` coverage nir
        * `2**10` coverage red
        * `2**11` coverage blue
        """
        if slice_ is None:
            slice_ = (slice(None), slice(None))

        with h5py.File(self.hdf5_file, "r") as input_f:
            quality = input_f['LEVEL2A/QUALITY/SM'][slice_]
        return quality

    def load_mask(self, slice_=None):
        return mask_only_sm(self.load_sm(slice_=slice_))


def sm_cloud_mask(sm, mask_undefined=False):
    """
    Returns a binary cloud mask: 1 cloudy values 0 rest

    From user manual http://www.vito-eodata.be/PDF/image/PROBAV-Products_User_Manual.pdf pag 67
        * Clear  ->    000
        * Shadow ->    001
        * Undefined -> 010
        * Cloud  ->    011
        * Ice    ->    100

    :param sm: sm flags as loaded from ProbaVImageOperational.load_sm() method
    :param mask_undefined: True returns np.ma.masked_array with undefined values masked
    :return:
    """
    cloud_mask = np.uint8((sm & 1 != 0) & (sm & 2**1 != 0) & (sm & 2**2 == 0))
    if mask_undefined:
        undefined_mask = (sm & 1 == 0) & (sm & 2**1 != 0) & (sm & 2**2 == 0)
        cloud_mask = np.ma.masked_array(cloud_mask,undefined_mask)

    return cloud_mask


def mask_only_sm(sm):
    mascara = np.zeros(sm.shape, dtype=np.bool)
    for i in range(4):
        mascara |= ((sm & (2 ** (i + 8))) == 0)

    return mascara


def mask_operational(bands, sm):
    # http://www.vito-eodata.be/PDF/image/PROBAV-Products_User_Manual.pdf
    mascara = np.any(bands < 0, axis=-1)
    mascara |= mask_only_sm(sm)

    return mascara



