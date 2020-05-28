import tensorflow as tf
import random
import h5py


def get_dataset_inmemory(path_cache):

    with h5py.File(path_cache, "r") as dat:
        pvdata = dat["PV"][...]
        ludata = dat["LU"][...]
        len_ds = pvdata.shape[0]
        tuple_datasets = (pvdata,  ludata)

    datafinal = tf.data.Dataset.from_tensor_slices(tuple_datasets)
    return datafinal.shuffle(buffer_size=1500), len_ds


def d4_data_augmentation(dataset):
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        r = tf.image.random_flip_left_right(r, seed=seed)
        r = tf.image.random_flip_up_down(r, seed=seed)

        k = tf.random.uniform([],
                              dtype=tf.int32,
                              minval=0,
                              maxval=4,
                              seed=seed)

        r = tf.image.rot90(r,k=k)

        return r

    return dataset.map(lambda *args: tuple(transform(a) for a in args))


def make_batches(ds,
                 batch_size=32,
                 data_augmentation_fun=None,
                 drop_remainder=True,
                 repeat=True):

    if data_augmentation_fun is not None:
        ds = data_augmentation_fun(ds)

    if repeat:
        ds = ds.repeat()

    return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(None)