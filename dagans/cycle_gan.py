import tensorflow as tf
from dagans import gan_model
from tqdm import tqdm
import datetime
import logging
import time


def kl_divergence(y_true, y_pred):
    EPS = 1e-6
    y_true = tf.clip_by_value(y_true, EPS, 1 - EPS)
    y_pred = tf.clip_by_value(y_pred, EPS, 1 - EPS)
    kl_div = y_true * tf.math.log(y_true/y_pred) + (1 - y_true) * tf.math.log((1 - y_true)/(1 - y_pred))
    return tf.reduce_mean(kl_div)


class CycleGAN:
    def __init__(self, genpvl8, genl8pv, discpv, discl8, l8_cloud_model,gradient_penalty_fun):
        self.genpvl8 = genpvl8
        self.genl8pv = genl8pv
        self.discpv = discpv
        self.discl8 = discl8
        self.l8_cloud_model = l8_cloud_model
        self.gradient_penalty_fun= gradient_penalty_fun

    def train_step_fun(self, lr=0.0001, beta1=.5, identity_lmbda=5, cycled_lmbda=5,
                       segmentation_consistency_lmbda=1, gradient_penalty_weight=0.):
        genpvl8_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta1)
        genl8pv_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta1)
        discpv_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta1)
        discl8_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta1)

        # assert (cycled_cloud_consistency_lmbda == 0) or (self.l8_cloud_model is not None), \
        #     "Cloud consistency lambda different than 0 but not cloud detection dagans provided"

        object_cyclegan = self

        def train_step(pv_image, l8_image):
            with tf.GradientTape(persistent=True) as tape:
                l8_fake_image = object_cyclegan.genpvl8(pv_image, training=True)
                pv_fake_image = object_cyclegan.genl8pv(l8_image, training=True)
                cycled_l8_image = object_cyclegan.genpvl8(pv_fake_image, training=True)
                cycled_pv_image = object_cyclegan.genl8pv(l8_fake_image, training=True)

                concat_l8images = tf.concat([l8_image, l8_fake_image], axis=0)
                disc_concat_output_l8 = object_cyclegan.discl8(concat_l8images,
                                                            training=True)

                disc_l8real = disc_concat_output_l8[:l8_image.shape[0]]
                disc_l8fake = disc_concat_output_l8[l8_image.shape[0]:]

                concat_pvimages = tf.concat([pv_image, pv_fake_image], axis=0)
                disc_concat_output_pv = object_cyclegan.discpv(concat_pvimages,
                                                               training=True)

                disc_pvreal = disc_concat_output_pv[:pv_image.shape[0]]
                disc_pvfake = disc_concat_output_pv[pv_image.shape[0]:]

                gen_pvl8_loss = gan_model.generator_gan_loss(disc_l8fake)

                disc_l8_loss = gan_model.discriminator_loss(disc_l8real, disc_l8fake)
                gp_l8_loss = object_cyclegan.gradient_penalty_fun(object_cyclegan.discl8, l8_image, pv_fake_image)
                disc_l8_total_loss = disc_l8_loss + gradient_penalty_weight * gp_l8_loss

                gen_l8pv_loss = gan_model.generator_gan_loss(disc_pvfake)
                disc_pv_loss = gan_model.discriminator_loss(disc_pvreal, disc_pvfake)
                gp_pv_loss = object_cyclegan.gradient_penalty_fun(object_cyclegan.discpv, pv_image, l8_fake_image)
                disc_pv_total_loss = disc_pv_loss + gradient_penalty_weight * gp_pv_loss

                identity_loss_pvl8 = gan_model.mae_loss(pv_image, l8_fake_image)

                identity_loss_l8pv = gan_model.mae_loss(l8_image, pv_fake_image)

                cycle_loss_l8 = gan_model.mae_loss(l8_image, cycled_l8_image)
                cycle_loss_pv = gan_model.mae_loss(pv_image, cycled_pv_image)

                clouds_l8 = object_cyclegan.l8_cloud_model(l8_image, training=False)

                clouds_pv = object_cyclegan.l8_cloud_model(pv_image, training=False)
                clouds_pv_fake = object_cyclegan.l8_cloud_model(pv_fake_image, training=False)
                clouds_l8_fake = object_cyclegan.l8_cloud_model(l8_fake_image, training=False)


                loss_cloud_consistency_pv2l8 = kl_divergence(clouds_pv, clouds_l8_fake)
                loss_cloud_consistency_l82pv = kl_divergence(clouds_l8, clouds_pv_fake)

                total_consistency_gen_loss = loss_cloud_consistency_pv2l8 + loss_cloud_consistency_l82pv

                total_cycle_loss = cycle_loss_l8 + cycle_loss_pv

                gen_pvl8_total_loss = gen_pvl8_loss + identity_lmbda * identity_loss_pvl8 + \
                                      cycled_lmbda * total_cycle_loss +\
                                      segmentation_consistency_lmbda * loss_cloud_consistency_pv2l8
                gen_l8pv_total_loss = gen_l8pv_loss + identity_lmbda * identity_loss_l8pv + \
                                      cycled_lmbda * total_cycle_loss +\
                                      segmentation_consistency_lmbda * loss_cloud_consistency_l82pv

            genpvl8_gradients = tape.gradient(gen_pvl8_total_loss,
                                              object_cyclegan.genpvl8.trainable_variables)
            genl8pv_gradients = tape.gradient(gen_l8pv_total_loss,
                                              object_cyclegan.genl8pv.trainable_variables)

            discpv_gradients = tape.gradient(disc_pv_total_loss,
                                             object_cyclegan.discpv.trainable_variables)
            discl8_gradients = tape.gradient(disc_l8_total_loss,
                                             object_cyclegan.discl8.trainable_variables)

            genpvl8_optimizer.apply_gradients(zip(genpvl8_gradients,
                                                  object_cyclegan.genpvl8.trainable_variables))
            genl8pv_optimizer.apply_gradients(zip(genl8pv_gradients,
                                                  object_cyclegan.genl8pv.trainable_variables))
            discpv_optimizer.apply_gradients(zip(discpv_gradients,
                                                 object_cyclegan.discpv.trainable_variables))
            discl8_optimizer.apply_gradients(zip(discl8_gradients,
                                                 object_cyclegan.discl8.trainable_variables))

            return gen_pvl8_loss, gen_l8pv_loss, identity_loss_pvl8, identity_loss_l8pv, \
                   cycle_loss_pv, cycle_loss_l8, disc_pv_loss, disc_l8_loss, \
                   gp_pv_loss, gp_l8_loss, \
                   total_consistency_gen_loss, \
                   gen_pvl8_total_loss, gen_l8pv_total_loss

        return tf.function(train_step)


def fit(train_ds, epochs, obj_cyclegan, steps_per_epoch=None, frec_update=1000, val_ds=None, lr=0.0001,
        logdir=None, identity_lmbda=5, cycled_lmbda=5, segmentation_consistency_lmbda=1, gradient_penalty_weight=0):

    train_step = obj_cyclegan.train_step_fun(lr=lr, identity_lmbda=identity_lmbda, cycled_lmbda=cycled_lmbda,
                                             segmentation_consistency_lmbda=segmentation_consistency_lmbda,
                                             gradient_penalty_weight=gradient_penalty_weight)

    pbar = tqdm(total=epochs)
    steps_per_epoch_str = "undef" if steps_per_epoch is None else str(steps_per_epoch)

    if val_ds is None:
        val_ds = train_ds
    if logdir is None:
        logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    summary_writer = tf.summary.create_file_writer(logdir)
    logging.info("Run:\n tensorboard --logdir %s" % logdir)

    total = tf.cast(0,"int64")

    for epoch in range(epochs):
        start = time.time()

        for pv_image, l8_image in val_ds.take(1):

            l8_fake = obj_cyclegan.genpvl8(pv_image)
            pv_cycled = obj_cyclegan.genl8pv(l8_fake)
            pv_fake = obj_cyclegan.genl8pv(l8_image)
            l8_cycled = obj_cyclegan.genpvl8(pv_fake)
            with summary_writer.as_default():
                tf.summary.image("PV Image", gan_model.probav_rgb(pv_image), step=total)
                tf.summary.image("L8 Fake", gan_model.probav_rgb(l8_fake), step=total)
                tf.summary.image("PV Cycled", gan_model.probav_rgb(pv_cycled), step=total)
                tf.summary.image("L8 Image", gan_model.probav_rgb(l8_image), step=total)
                tf.summary.image("PV Fake", gan_model.probav_rgb(pv_fake), step=total)
                tf.summary.image("L8 Cycled", gan_model.probav_rgb(l8_cycled), step=total)

        # Train
        for n, (pv_image, l8_image) in train_ds.enumerate():
            # tf.summary.trace_on(graph=True)
            gen_pvl8_loss, gen_l8pv_loss, identity_loss_pvl8, identity_loss_l8pv, \
            cycle_loss_pv, cycle_loss_l8, disc_pv_loss, disc_l8_loss, \
            gp_pv_loss, gp_l8_loss, \
            total_consistency_gen_loss, \
            gen_pvl8_total_loss, gen_l8pv_total_loss = train_step(pv_image, l8_image)

            with summary_writer.as_default():
                tf.summary.scalar('gen_pvl8_loss', gen_pvl8_loss, step=total)
                tf.summary.scalar('gen_l8pv_loss', gen_l8pv_loss, step=total)
                tf.summary.scalar('identity_loss_pvl8', identity_loss_pvl8, step=total)
                tf.summary.scalar('identity_loss_l8pv', identity_loss_l8pv, step=total)
                tf.summary.scalar('cycle_loss_pv', cycle_loss_pv, step=total)
                tf.summary.scalar('cycle_loss_l8', cycle_loss_l8, step=total)
                tf.summary.scalar('total_consistency_gen_loss', total_consistency_gen_loss, step=total)
                tf.summary.scalar('gen_pvl8_total_loss', gen_pvl8_total_loss, step=total)
                tf.summary.scalar('gen_l8pv_total_loss', gen_l8pv_total_loss, step=total)
                tf.summary.scalar('disc_pv_loss', disc_pv_loss, step=total)
                tf.summary.scalar('disc_l8_loss', disc_l8_loss, step=total)
                tf.summary.scalar('gp_pv_loss', gp_pv_loss, step=total)
                tf.summary.scalar('gp_l8_loss', gp_l8_loss, step=total)

            total += 1
            if (n % frec_update) == 0:
                pbar.set_description('Epoch %d step %d/%s: last batch PV->L8 loss = %.4f\t L8->PV loss: %.4f Disc PV loss: %.4f Disc L8 loss: %.4f' %
                                     (epoch, int(n),steps_per_epoch_str, float(gen_pvl8_total_loss),float(gen_l8pv_total_loss),
                                      float(disc_pv_loss), float(disc_l8_loss)))

        time_taken = time.time()-start
        pbar.update(1)
        pbar.write("Epoch: %d. Time took in %d steps %.0f(s)"%(epoch, int(n), time_taken))

    logging.info("Train finished. Run:\n tensorboard --logdir %s" % logdir)
