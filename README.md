# BWGAN
Code for the paper [Banach Wasserstein GAN](https://arxiv.org/abs/1806.06621).

# Description
Traditional [WGAN](https://arxiv.org/abs/1701.07875) uses an approximation of the Wasserstein metric to opimize the generator. This Wasserstein metric in turn depends upon an underlying metric on _images_ which is taken to be the <img src="https://latex.codecogs.com/svg.latex?%5Cell%5E2"> norm

<img src="https://latex.codecogs.com/svg.latex?%5C%7Cx%5C%7C_%7B2%7D%20%3D%20%5Cleft%28%20%5Csum_%7Bi%3D1%7D%5En%20x_i%5E2%20%5Cright%29%5E%7B1/2%7D">

The article extends the theory of [WGAN-GP](https://arxiv.org/abs/1704.00028) to any [Banach space](https://en.wikipedia.org/wiki/Banach_space), while this code can be used to train WGAN over any [Sobolev space](https://en.wikipedia.org/wiki/Sobolev_space) <img src="https://latex.codecogs.com/svg.latex?W%5E%7Bs%2C%20p%7D"> with norm

<img src="https://latex.codecogs.com/svg.latex?%5C%7Cf%5C%7C_%7BW%5E%7Bs%2C%20p%7D%7D%20%3D%20%5Cleft%28%20%5Cint_%7B%5COmega%7D%20%5Cleft%28%20%5Cmathcal%7BF%7D%5E%7B-1%7D%20%5Cleft%5B%20%281%20&plus;%20%7C%5Cxi%7C%5E2%29%5E%7Bs/2%7D%20%5Cmathcal%7BF%7D%20f%20%5Cright%5D%28x%29%20%5Cright%29%5Ep%20dx%20%5Cright%29%5E%7B1/p%7D">

The parameters _p_ can be used to control the focus on outliers, with high _p_ indicating a strong focus on the worst offenders. _s_ can be used to control focus on small/large scale behaviour, where negative _s_ indicates focus on large scales, while positive _s_ indicates focus on small scales (e.g. edges).

# Results

Inception scores for the spaces <img src="https://latex.codecogs.com/svg.latex?W%5E%7Bs%2C%202%7D"> and <img src="https://latex.codecogs.com/svg.latex?W%5E%7B0%2C%20p%7D">:

<img src="https://user-images.githubusercontent.com/2202312/40432213-7603b084-5ea9-11e8-894d-b28db89ba1b3.png" width="350" title="Github Logo"> <img src="https://user-images.githubusercontent.com/2202312/40432231-7de2a148-5ea9-11e8-87ec-979b090ab1f2.png" width="350" title="Github Logo">


Dependencies
------------
The code has some dependencies that can be easily installed

```bash
$ pip install https://github.com/adler-j/tensordata/archive/master.zip
$ pip install https://github.com/adler-j/adler/archive/master.zip
```

You also need a recent version of tensorflow in order to use the `tf.contrib.gan` functionality.
