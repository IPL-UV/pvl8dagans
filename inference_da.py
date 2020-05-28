import luigi
from dagans import gan_model, cloud_model
from utils import probav_image_operational
from utils import predbytiles
import numpy as np
import shutil
import h5py


class InferenceDA(luigi.Task):
    """
    Test the proposed DA method on a given Proba-V image.

    python inference_da.py InferenceDA --pvimage /path/to/pvimage.HDF5 --image-dest /path/save/img.HDF5

    """
    pvimage = luigi.Parameter(description="Proba-V image to do inference")
    image_dest = luigi.Parameter(description="destination file")
    cloud_detection_weights = luigi.Parameter(default="checkpoints/cloud_detection_l8.hdf5")
    dagans_weights = luigi.Parameter(default="checkpoints/full.hdf5")

    def output(self):
        return luigi.LocalTarget(self.image_dest)

    def run(self):
        genpv2l8 = gan_model.generator_simple((None, None, 4), df=64,
                                              normtype="batchnorm")

        genpv2l8.load_weights(self.dagans_weights)
        model_clouds = cloud_model.load_model((None, None), weight_decay=0)
        model_clouds.load_weights(self.cloud_detection_weights)

        pvimage = probav_image_operational.ProbaVImageOperational(self.pvimage)

        pvasl8image = predbytiles.predict(lambda x: genpv2l8.predict(x[np.newaxis], batch_size=1)[0], pvimage)

        # Copy pvimage to image dest
        shutil.copy(self.pvimage, self.image_dest)

        # Modify RGB bands from copied image
        pvimage_dest = probav_image_operational.ProbaVImageOperational(self.image_dest)
        pvimage_dest.save_bands(pvasl8image)

        # Compute cloud mask from transformed image
        clouds_pvimage = predbytiles.predict(predbytiles.padded_predict(model_clouds, 4), pvimage_dest)

        # Save cloud mask
        with h5py.File(pvimage_dest.hdf5_file, "r+") as input_f:
            input_f.create_dataset("CM_PVDAGANS", data=clouds_pvimage, compression="gzip")


if __name__ == "__main__":
    luigi.run(local_scheduler=True)







