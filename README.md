# Modular Flow Matching

<p align="right">
English | <a href="README.zh-CN.md">中文</a>
</p>

A compact Flow Matching playground that maps Rectified Flow, conditional
velocity prediction, and ODE sampling to small, readable PyTorch modules.

```text
Gaussian noise (+ optional class label) -> learned velocity field -> CIFAR10 image
```

The repository is built as a learning-oriented implementation: the probability
path, regression target, velocity backbone, and numerical sampler remain
visible and replaceable.

For setup, data preparation, training, and sampling commands, see
[Documentation](Documentation/README.md).

## The Idea

Flow Matching learns a continuous-time vector field that transports a simple
source distribution into the data distribution:

```math
\frac{d x_t}{d t} = v_\theta(x_t, t).
```

Generation starts from standard Gaussian noise and follows this learned ODE
from `t=0` to `t=1`:

```math
x_0 \sim \mathcal{N}(0, I),
\qquad
x_1 \sim p_{\mathrm{data}}.
```

The network predicts a velocity rather than a discrete reverse transition. The
sampler then decides how to integrate that velocity field.

At the distribution level, the path connects a Gaussian source to the data
distribution:

```math
p_0(x)=\mathcal{N}(0,I),
\qquad
p_1(x)=p_{\mathrm{data}}(x).
```

The evolving density and its velocity field satisfy the continuity equation:

```math
\frac{\partial p_t(x)}{\partial t}
+
\nabla\cdot\left(p_t(x)v_t(x)\right)
=0.
```

Probability mass is transported by the field rather than created or destroyed;
Flow Matching learns the field that produces this probability flow.

## Linear Probability Path

The simplest path independently pairs a noise sample `z` with a data sample
`x` and interpolates between them:

```math
z \sim \mathcal{N}(0, I),
\qquad
x \sim p_{\mathrm{data}}.
```

```math
x_t = (1-t)z + tx.
```

The endpoints are `x_0=z` and `x_1=x`. Differentiating the path gives the
conditional target velocity:

```math
u_t = \frac{d x_t}{d t} = x-z.
```

This straight-line construction is the simplest Rectified Flow case, with
`x_0=z`, `x_1=x`, and constant conditional velocity `x_1-x_0`.

Path construction is implemented in [`flow/paths.py`](flow/paths.py).

## The Training Objective

Each training step samples a real image, Gaussian noise, and a uniformly random
time. The network receives the interpolated point `x_t` and regresses the path
velocity:

```math
x\sim p_{\mathrm{data}},
\qquad
z\sim\mathcal{N}(0,I),
\qquad
t\sim U(0,1).
```

```math
\mathcal{L}_{\mathrm{FM}}
=
\mathbb{E}_{x,z,t}
\left[
\left\|
v_\theta(x_t,t) - (x-z)
\right\|^2
\right].
```

The target depends on the sampled pair `(x,z)`, while the model only observes
`(x_t,t)`. Under mean squared error, the optimal prediction is therefore the
conditional expectation:

```math
v^*(x_t,t)
=
\mathbb{E}
\left[
x-z \mid x_t,t
\right].
```

The loss wrapper lives in [`flow/losses.py`](flow/losses.py), and the training
loop calls it from [`train.py`](train.py).

## Sampling

Sampling starts from Gaussian noise and numerically integrates the learned
velocity field:

```math
\frac{d x_t}{d t}=v_\theta(x_t,t),
\qquad
t:0\rightarrow1.
```

This project supports Euler and Heun integration. Euler uses one velocity
evaluation per step, while Heun uses a predictor-corrector update for a more
accurate trajectory. More steps usually produce a smoother trajectory, while
fewer steps reduce generation time.

The ODE samplers are implemented in [`flow/sampler.py`](flow/sampler.py).

## Inside This Demo

The experiment uses normalized CIFAR10 images in pixel space:

```math
x \in \mathbb{R}^{3\times32\times32}.
```

The same UNet can be trained unconditionally or with a CIFAR10 class label:

```math
v_\theta(x_t,t),
\qquad
v_\theta(x_t,t,y).
```

For class-conditional training, a class embedding is added to the time
embedding and injected into the UNet residual blocks. Sampling can cycle class
labels, target one class, or generate a grid with one row per class.

| Component | Implementation |
| --- | --- |
| Linear probability path | [`flow/paths.py`](flow/paths.py) |
| Flow Matching loss | [`flow/losses.py`](flow/losses.py) |
| Euler and Heun samplers | [`flow/sampler.py`](flow/sampler.py) |
| Conditional UNet | [`models/unet.py`](models/unet.py) |
| Training entry point | [`train.py`](train.py) |
| Sampling entry point | [`sample.py`](sample.py) |

## Relation to Nearby Methods

| Method | Learned object | Generation |
| --- | --- | --- |
| VAE | latent posterior, prior, and decoder | sample a latent and decode it |
| Diffusion | noise, score, data, or velocity prediction | reverse SDE, ODE, or discrete denoising chain |
| Flow Matching | continuous vector field | solve an ODE from noise to data |
| This repository | pixel-space conditional velocity field | Euler or Heun integration on CIFAR10 |

Diffusion velocity prediction usually starts from a prescribed noise path:

```math
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad
v=\alpha_t\epsilon-\sigma_t x_0.
```

It is a prediction-target parameterization inside the diffusion framework. For
the straight Flow Matching path used here, velocity is instead the direct time
derivative of the probability path:

```math
x_t=(1-t)z+tx,
\qquad
v=x-z.
```

## Results

The images below are qualitative samples from the completed class-conditional
CIFAR10 run in [`assets/results/`](assets/results).

Training and full-pipeline scripts write generated samples to:

```text
runs/cifar10_fm/samples/
runs/full_pipeline/conditional/samples/
runs/full_pipeline/unconditional/samples/
```

### Class-Conditional Grid

Each row corresponds to one CIFAR10 class.

<img src="assets/results/class_grid_euler_050.png" alt="Class-conditional CIFAR10 Flow Matching samples" width="520">

### Training Sample at Epoch 100

<img src="assets/results/epoch_0100.png" alt="CIFAR10 Flow Matching samples at epoch 100" width="520">

Common generated artifacts are:

| Result | Purpose | Default filename |
| --- | --- | --- |
| Class grid | One row per CIFAR10 class | `class_grid_euler_050.png` |
| Class cycle | Quick check of conditional control | `class_cycle_euler_050.png` |
| Single class | Samples for one class, such as `cat` | `class_3_euler_050.png` |
| Unconditional | Samples from the unconditional model | `samples_euler_050.png` |
| Trajectory | ODE evolution from noise to image | `*.gif` |

## Takeaway

Flow Matching separates three ideas: a probability path defines how noise and
data are connected, the network learns the path velocity, and an ODE solver
turns that velocity field into generated samples.

A VAE samples a latent and decodes it; diffusion learns a reverse denoising
process; Flow Matching learns the continuous field that transports noise into
data.

```math
x_t=(1-t)z+tx,
\qquad
u_t=x-z.
```

```math
\mathrm{sample}
=
\mathrm{ODESolver}\left(v_\theta,x_0\right),
\qquad
x_0\sim\mathcal{N}(0,I).
```
