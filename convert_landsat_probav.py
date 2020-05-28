import luigi
import os
from utils import landsat_as_pv
from utils import l8image


class ConvertPV(luigi.Task):
    """
    Convert Landsat-8 data from Biome or 38-Clouds dataset to same spectral and spatial resolution as Proba-V.
    Following the transformation proposed in:
    Transferring deep learning models for cloud detection between Landsat-8 and Proba-V
    https://www.sciencedirect.com/science/article/abs/pii/S0924271619302801

    Landsat-8 file is exported in the same format as Proba-V (HDF5 files)

    python convert_landsat_probav.py ConvertPV --l8img BC/LC80010112014080LGN00

    """
    l8img = luigi.Parameter(description="Folder with Landsat 8 image")
    outfolder = luigi.Parameter(default="landsataspv")
    resolution = luigi.ChoiceParameter(choices=["333M", "100M", "1KM"], default="333M")
    type_product = luigi.ChoiceParameter(choices=["biome", "38c", "landsat8"], default="landsat8",
                                         description="Flag that indicates if the product has a manually annotated "
                                                     "cloud mask from Biome or 38-Clouds dataset")

    def l8obj(self):
        if not hasattr(self, "l8obj_computed"):
            if self.type_product == "biome":
                obj = l8image.Biome(self.l8img)
            elif self.type_product == "38c":
                obj = l8image.L8_38Clouds(self.l8img)
            else:
                obj = l8image.L8Image(self.l8img)

            setattr(self, "l8obj_computed",
                    obj)
        return getattr(self, "l8obj_computed")

    def output(self):
        l8img = self.l8obj()
        path_out = os.path.join(self.outfolder,
                                l8img.name,
                                l8img.name+"_"+str(self.resolution)+".HDF5")
        return luigi.LocalTarget(path_out)

    def run(self):
        l8img = self.l8obj()
        out = self.output()
        out.makedirs()
        landsat_as_pv.convert_landsat_to_probav(l8img, self.resolution,
                                                file_out=out.path)


if __name__ == "__main__":
    luigi.run(local_scheduler=True)
