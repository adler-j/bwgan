"""Utilities for computing the sobolev norm."""

import tensorflow as tf

def sobolev_filter(x, c=5, s=1):
    """Apply sobolev filter.

    Parameters
    ----------
    x : tensorflow.Tensor of shape B W H C
        txt
    c : float
        Scaling of the cooridinate systems (1 / pixel size)
    s : float
        Order of the Sobolev norm
    """
    with tf.name_scope('sobolev'):
        # FFT is taken over the innermost axes, so move channel to beginning.
        x = tf.transpose(x, [0, 3, 1, 2])
        fft_x = tf.spectral.fft2d(tf.cast(x, 'complex64'))

        shape = tf.shape(fft_x)
        sx = shape[3]
        sy = shape[2]

        # Construct meshgrid for the scale
        x = tf.range(sx)
        x = tf.minimum(x, sx - x)
        x = tf.cast(x, dtype='complex64') / tf.cast(sx // 2, dtype='complex64')
        y = tf.range(sy)
        y = tf.minimum(y, sy - y)
        y = tf.cast(y, dtype='complex64') / tf.cast(sy // 2, dtype='complex64')
        X, Y = tf.meshgrid(x, y)
        X = X[None, None]
        Y = Y[None, None]

        scale = (1 + c * (X ** 2 + Y ** 2)) ** (s / 2)

        # Compute spatial gradient in fourier space
        fft_x = scale * fft_x

        result_x = tf.spectral.ifft2d(fft_x)
        result_x = tf.real(result_x)
        return tf.transpose(result_x, [0, 2, 3, 1])


if __name__ == '__main__':
    import scipy
    import numpy as np
    import matplotlib.pyplot as plt

    img = scipy.misc.face()[None, ...] / 256.0
    print(img.shape)

    sess = tf.InteractiveSession()

    img_tf = tf.image.resize_bilinear(img, [32, 32])
    result_tf = sobolev_filter(img_tf, c=5.0, s=1)
    result = sess.run(result_tf)
    inp = sess.run(img_tf)

    plt.figure('input')
    plt.imshow(inp[0, ..., 0])
    plt.colorbar()

    plt.figure('result')
    result_show = result[0]
    plt.imshow(result_show[..., 0])
    plt.colorbar()

    plt.figure('diff')
    diff = inp[0] - result[0]
    plt.imshow(diff[..., 0])
    plt.colorbar()