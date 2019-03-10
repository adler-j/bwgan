"""Code for training Banach Wasserstein GAN on CIFAR 10.

With all the dependencies installed, the code should run as-is. 
Data is downloaded on the fly.
"""

import adler.tensorflow as atf
import sys
import tensorflow as tf
import numpy as np
import tensordata
import functools
from sobolev_utils import sobolev_filter

# User selectable parameters
EXPONENT = 2
SOBOLEV_C = 5.0
SOBOLEV_S = 0
MAX_ITERS = 100000
SUMMARY_FREQ = 10
INCEPTION_FREQ = 1000
BATCH_SIZE = 64
BATCH_SIZE_TEST = 100
reset = True

# set seeds for reproducibility
np.random.seed(0)
tf.set_random_seed(0)

sess = tf.InteractiveSession()

# Training specific parameters
name = 'cifar10_sobolev_5/p={}s={}'.format(EXPONENT, SOBOLEV_S)
size = 32
DUAL_EXPONENT = 1 / (1 - 1/EXPONENT) if EXPONENT != 1 else np.inf

print('index={}, s={}, p={}, q={}'.format(index, SOBOLEV_S, EXPONENT, DUAL_EXPONENT))

with tf.name_scope('placeholders'):
    x_train_ph, _ = tensordata.get_cifar10_tf(batch_size=BATCH_SIZE)
    x_test_ph, _ = tensordata.get_cifar10_tf(batch_size=BATCH_SIZE_TEST)

    is_training = tf.placeholder(bool, name='is_training')
    use_agumentation = tf.identity(is_training, name='is_training')


with tf.name_scope('pre_process'):
    x_train = (x_train_ph - 0.5) * 2.0
    x_test = (x_test_ph - 0.5) * 2.0

    x_true = tf.cond(is_training,
                     lambda: x_train,
                     lambda: x_test)

def apply_conv(x, filters=32, kernel_size=3, he_init=True):
    if he_init:
        initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True)
    else:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)

    return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                            padding='SAME', kernel_initializer=initializer)


def activation(x):
    with tf.name_scope('activation'):
        return tf.nn.relu(x)


def bn(x):
    return tf.contrib.layers.batch_norm(x,
                                    decay=0.9,
                                    center=True,
                                    scale=True,
                                    epsilon=1e-5,
                                    zero_debias_moving_mean=True,
                                    is_training=is_training)


def stable_norm(x, ord):
    x = tf.contrib.layers.flatten(x)
    alpha = tf.reduce_max(tf.abs(x) + 1e-5, axis=1)
    result = alpha * tf.norm(x / alpha[:, None], ord=ord, axis=1)
    return result


def downsample(x):
    with tf.name_scope('downsample'):
        x = tf.identity(x)
        return tf.add_n([x[:,::2,::2,:], x[:,1::2,::2,:],
                         x[:,::2,1::2,:], x[:,1::2,1::2,:]]) / 4.

def upsample(x):
    with tf.name_scope('upsample'):
        x = tf.identity(x)
        x = tf.concat([x, x, x, x], axis=-1)
        return tf.depth_to_space(x, 2)


def conv_meanpool(x, **kwargs):
    return downsample(apply_conv(x, **kwargs))

def meanpool_conv(x, **kwargs):
    return apply_conv(downsample(x), **kwargs)

def upsample_conv(x, **kwargs):
    return apply_conv(upsample(x), **kwargs)

def resblock(x, filters, resample=None, normalize=False):
    if normalize:
        norm_fn = bn
    else:
        norm_fn = tf.identity

    if resample == 'down':
        conv_1 = functools.partial(apply_conv, filters=filters)
        conv_2 = functools.partial(conv_meanpool, filters=filters)
        conv_shortcut = functools.partial(conv_meanpool, filters=filters,
                                          kernel_size=1, he_init=False)
    elif resample == 'up':
        conv_1 = functools.partial(upsample_conv, filters=filters)
        conv_2 = functools.partial(apply_conv, filters=filters)
        conv_shortcut = functools.partial(upsample_conv, filters=filters,
                                          kernel_size=1, he_init=False)
    elif resample == None:
        conv_1 = functools.partial(apply_conv, filters=filters)
        conv_2 = functools.partial(apply_conv, filters=filters)
        conv_shortcut = tf.identity

    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = conv_1(activation(norm_fn(x)))
        update = conv_2(activation(norm_fn(update)))

        skip = conv_shortcut(x)
        return skip + update


def resblock_optimized(x, filters):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        update = conv_meanpool(activation(update), filters=filters)

        skip = meanpool_conv(x, filters=filters, kernel_size=1, he_init=False)
        return skip + update


def generator(z, reuse):
    with tf.variable_scope('generator', reuse=reuse):
        with tf.name_scope('pre_process'):
            z = tf.layers.dense(z, 4 * 4 * 128)
            x = tf.reshape(z, [-1, 4, 4, 128])

        with tf.name_scope('x1'):
            x = resblock(x, filters=128, resample='up', normalize=True) # 8
            x = resblock(x, filters=128, resample='up', normalize=True) # 16
            x = resblock(x, filters=128, resample='up', normalize=True) # 32

        with tf.name_scope('post_process'):
            x = activation(bn(x))
            result = apply_conv(x, filters=3, he_init=False)
            return tf.tanh(result)


def discriminator(x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.name_scope('pre_process'):
            x = resblock_optimized(x, filters=128)

        with tf.name_scope('x1'):
            x = resblock(x, filters=128, resample='down') # 8
            x = resblock(x, filters=128) # 16
            x = resblock(x, filters=128) # 32

        with tf.name_scope('post_process'):
            x = activation(x)
            x = tf.reduce_mean(x, axis=[1, 2])
            flat = tf.contrib.layers.flatten(x)
            flat = tf.layers.dense(flat, 1)
            return flat


with tf.name_scope('gan'):
    z = tf.random_normal([tf.shape(x_true)[0], 128], name="z")

    x_generated = generator(z, reuse=False)

    d_true = discriminator(x_true, reuse=False)
    d_generated = discriminator(x_generated, reuse=True)

    z_gen = tf.random_normal([BATCH_SIZE * 2, 128], name="z")
    d_generated_train = discriminator(generator(z_gen, reuse=True), reuse=True)

with tf.name_scope('dual_norm'):
    sobolev_true = sobolev_filter(x_true, c=SOBOLEV_C, s=SOBOLEV_S)
    lamb = tf.reduce_mean(stable_norm(sobolev_true, ord=EXPONENT))
    dual_sobolev_true = sobolev_filter(x_true, c=SOBOLEV_C, s=-SOBOLEV_S)
    gamma = tf.reduce_mean(stable_norm(sobolev_true, ord=DUAL_EXPONENT))

with tf.name_scope('regularizer'):
    epsilon = tf.random_uniform([tf.shape(x_true)[0], 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * x_generated + (1 - epsilon) * x_true
    d_hat = discriminator(x_hat, reuse=True)

    gradients = tf.gradients(d_hat, x_hat)[0]
    dual_sobolev_gradients = sobolev_filter(gradients, c=SOBOLEV_C, s=-SOBOLEV_S)
    ddx = stable_norm(dual_sobolev_gradients, ord=DUAL_EXPONENT)

    d_regularizer = tf.reduce_mean(tf.square(ddx / gamma - 1))
    d_regularizer_mean = tf.reduce_mean(tf.square(d_true))

with tf.name_scope('loss_gan'):
    wasserstein_scaled = (tf.reduce_mean(d_generated) - tf.reduce_mean(d_true))
    wasserstein = wasserstein_scaled / gamma

    g_loss = tf.reduce_mean(d_generated_train) / gamma
    d_loss = (-wasserstein +
              lamb * d_regularizer +
              1e-5 * d_regularizer_mean)

with tf.name_scope('optimizer'):
    ema = atf.EMAHelper(decay=0.99)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    decay = tf.maximum(0., 1.-(tf.cast(global_step, tf.float32)/MAX_ITERS))
    learning_rate = 2e-4 * decay
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0., beta2=0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/generator')
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    with tf.control_dependencies(update_ops):
        g_train = optimizer.minimize(g_loss, var_list=g_vars,
                                     global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/discriminator')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    with tf.control_dependencies(update_ops):
        d_train = optimizer.minimize(d_loss, var_list=d_vars)


with tf.name_scope('summaries'):
    tf.summary.scalar('wasserstein_scaled', wasserstein_scaled)
    tf.summary.scalar('wasserstein', wasserstein)

    tf.summary.scalar('g_loss', g_loss)

    tf.summary.scalar('d_loss', d_loss)
    atf.scalars_summary('d_true', d_true)
    atf.scalars_summary('d_generated', d_generated)
    tf.summary.scalar('d_regularizer', d_regularizer)
    tf.summary.scalar('d_regularizer_mean', d_regularizer_mean)

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('global_step', global_step)

    atf.scalars_summary('x_generated', x_generated)
    atf.scalars_summary('x_true', x_true)

    atf.scalars_summary('gamma', gamma)
    atf.scalars_summary('lamb', lamb)

    atf.image_grid_summary('x_true', x_true)
    atf.image_grid_summary('x_generated', x_generated)
    atf.image_grid_summary('gradients', gradients)
    atf.image_grid_summary('dual_sobolev_gradients', dual_sobolev_gradients)

    atf.scalars_summary('ddx', ddx)
    atf.scalars_summary('gradients', gradients)
    atf.scalars_summary('dual_sobolev_gradients', dual_sobolev_gradients)

    merged_summary = tf.summary.merge_all()

    # Advanced metrics
    with tf.name_scope('inception'):
        # Specific function to compute inception score for very large
        # number of samples
        def generate_and_classify(z):
            INCEPTION_OUTPUT = 'logits:0'
            x = generator(z, reuse=True)
            x = tf.image.resize_bilinear(x, [299, 299])
            return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_OUTPUT)

        # Fixed z for fairness between runs
        inception_z = tf.constant(np.random.randn(10000, 128), dtype='float32')
        inception_score = tf.contrib.gan.eval.classifier_score(inception_z,
                                                               classifier_fn=generate_and_classify,
                                                               num_batches=10000 // 100)

        inception_summary = tf.summary.merge([
                tf.summary.scalar('inception_score', inception_score)])

        full_summary = tf.summary.merge([merged_summary, inception_summary])

    test_summary_writer, train_summary_writer = atf.util.summary_writers(name, cleanup=reset, write_graph=False)

# Initialize all TF variables
sess.run([tf.global_variables_initializer(),
          tf.local_variables_initializer()])

# Coordinate the loading of image files.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

# Add op to save and restore
saver = tf.train.Saver()
if not reset:
    saver.restore(sess,
                  atf.util.default_checkpoint_path(name))

# Standardized validation z
z_validate = np.random.randn(BATCH_SIZE_TEST, 128)

# Train the network
while True:
    i = sess.run(global_step)
    if i >= MAX_ITERS:
        break

    num_d_train = 5
    for j in range(num_d_train):
        _, d_loss_result = sess.run([d_train, d_loss],
                                    feed_dict={is_training: True})

    _, g_loss_result, _ = sess.run([g_train, g_loss, ema.apply],
             feed_dict={is_training: True})

    print('s={}, i={}, j={}, d_loss={}, g_loss={}'.format(SOBOLEV_S, i, j,
                                                    d_loss_result,
                                                    g_loss_result))

    if i % SUMMARY_FREQ == SUMMARY_FREQ - 1:
        ema_dict = ema.average_dict()
        merged_summary_result_train = sess.run(merged_summary,
                                         feed_dict={is_training: False,
                                                    **ema_dict})
        train_summary_writer.add_summary(merged_summary_result_train, i)
    if i % INCEPTION_FREQ == INCEPTION_FREQ - 1:
        ema_dict = ema.average_dict()
        merged_summary_result_test = sess.run(full_summary,
                                         feed_dict={z: z_validate,
                                                    is_training: False,
                                                    **ema_dict})
        test_summary_writer.add_summary(merged_summary_result_test, i)


    if i % 1000 == 999:
        saver.save(sess,
                   atf.util.default_checkpoint_path(name))
