import luigi
import os
from dagans import cycle_gan
from dagans import gan_model, cloud_model, dataloader
import tensorflow as tf
import numpy as np
import random
import logging

# Set seed for sanity
tf.random.set_seed(10)
np.random.seed(10)
random.seed(10)


FOLDER_MODELS = "run_checkpoints"
DATASETS_FOLDER_CACHE = "dataset/"


class TrainCycleGAN(luigi.Task):
    folder = luigi.Parameter(default="checkpoints_trained")
    dataset_folder = luigi.Parameter(description="Path to Biome Proba-V pseudo-simultaneous dataset")
    gradient_penalty_mode = luigi.ChoiceParameter(default="meschederreal", choices=["meschederreal", "none"])
    gradient_penalty_weight = luigi.FloatParameter(default=10.)
    df_gen = luigi.IntParameter(default=64)
    df_disc = luigi.IntParameter(default=8)
    epochs = luigi.IntParameter(default=25)
    seed = luigi.IntParameter(default=123)
    batch_size = luigi.IntParameter(default=48)
    lr = luigi.FloatParameter(default=1e-4)
    identity_lambda = luigi.FloatParameter(default=5.)
    cycled_lambda = luigi.FloatParameter(default=5.)
    segmentation_consistency_lmbda = luigi.FloatParameter(default=1.)
    weights_cloud_detection = luigi.Parameter(default="checkpoints/cnn_trainedbiome2.hdf5")
    suffix = luigi.Parameter(default="")
    normtype = luigi.ChoiceParameter(choices=["batchnorm", "instancenorm", "no"],
                                     default="batchnorm")

    def experiment_name(self):
        if self.suffix != "":
            suffix = "_"+self.suffix
        else:
            suffix = ""
        return "cycle_%d_%d%s" % (self.df_gen, self.df_disc, suffix)

    def output(self):
        genpvl8_name = luigi.LocalTarget(os.path.join(self.folder, "genpv2l8%s.hdf5" % self.experiment_name()))
        discl8_name = luigi.LocalTarget(os.path.join(self.folder, "discl8%s.hdf5" % self.experiment_name()))
        genl8pv_name = luigi.LocalTarget(os.path.join(self.folder, "genl82pv%s.hdf5" % self.experiment_name()))
        discpv_name = luigi.LocalTarget(os.path.join(self.folder, "discpv%s.hdf5" % self.experiment_name()))
        return [genpvl8_name, discl8_name, genl8pv_name, discpv_name]

    def run(self):
        # Set seed for sanity
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        shape = (64, 64)
        random.seed(self.seed)

        path_cache = os.path.join(self.dataset_folder, "simultaneous_landsatbiomeaspv_64_32.hdf5")
        assert os.path.exists(path_cache), "File %s does not exists" % path_cache

        dataset_train, len_dataset_train = dataloader.get_dataset_inmemory(path_cache)
        logging.info("Loaded dataset file %s. %d pseudo-simultaneous pairs" % (path_cache, len_dataset_train))

        batched_ds = dataloader.make_batches(dataset_train,
                                             data_augmentation_fun=dataloader.d4_data_augmentation,
                                             batch_size=self.batch_size,
                                             repeat=False)

        input_shape = shape + (4,)
        disc_l8 = gan_model.disc_model(input_shape, df=self.df_disc,
                                       activation=None)
        disc_pv = gan_model.disc_model(input_shape, df=self.df_disc,
                                       activation=None)

        genpv2l8 = gan_model.generator_simple(input_shape, df=self.df_gen,
                                              normtype=self.normtype)
        genl82pv = gan_model.generator_simple(input_shape, df=self.df_gen,
                                              normtype=self.normtype)

        # Load cloud detection dagans for seg loss
        cloud_detection_model = cloud_model.load_model(shape, weight_decay=0, final_activation="sigmoid")
        cloud_detection_model.load_weights(self.weights_cloud_detection)

        cyclegan = cycle_gan.CycleGAN(genl8pv=genl82pv, genpvl8=genpv2l8, discl8=disc_l8, discpv=disc_pv,
                                      l8_cloud_model=cloud_detection_model,
                                      gradient_penalty_fun=gan_model.gradient_penalty_fun(self.gradient_penalty_mode))

        steps_per_epoch = len_dataset_train // self.batch_size
        lr = self.lr

        cycle_gan.fit(train_ds=batched_ds, obj_cyclegan=cyclegan,
                      steps_per_epoch=steps_per_epoch, identity_lmbda=self.identity_lambda,
                      cycled_lmbda=self.cycled_lambda,
                      segmentation_consistency_lmbda=self.segmentation_consistency_lmbda,
                      gradient_penalty_weight=self.gradient_penalty_weight,
                      lr=lr, logdir=os.path.join(self.folder, "tflogs", self.experiment_name()),
                      epochs=self.epochs)

        outs = self.output()
        genpv2l8.save(outs[0].path)
        disc_l8.save(outs[1].path)
        genl82pv.save(outs[2].path)
        disc_pv.save(outs[3].path)


EXP_PAPER = {
    "fullrepr": {},
    "fullreprid0": {"identity_lambda": 0},
    "fullreprseg0": {"segmentation_consistency_lmbda": 0},
    "fullreprid0seg0": {"identity_lambda": 0, "segmentation_consistency_lmbda": 0},
    "fullreprcl0sl0": {"segmentation_consistency_lmbda": 0, "cycled_lambda": 0},
}


class TrainAllCycleGAN(luigi.WrapperTask):
    suffix = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=123)

    def requires(self):
        tasks = []
        for suffix, kwargs in EXP_PAPER.items():
            tasks.append(TrainCycleGAN(suffix=self.suffix+suffix, seed=self.seed, **kwargs))
        return tasks


if __name__ == "__main__":
    luigi.run(local_scheduler=True)
