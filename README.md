# Flow Matching on CIFAR-10

<p align="right">
English | <a href="README.zh-CN.md">中文</a>
</p>

This repository is a compact demonstration of **pixel-space Flow Matching / Rectified Flow** on CIFAR-10.

```text
Gaussian noise z (+ optional class label y) -> learned ODE flow -> CIFAR-10 image x
```

Flow Matching can be read as one simple idea:

> Learn a continuous-time velocity field that transports a simple noise distribution into the data distribution.

In this project, the simple distribution is a standard Gaussian, the data distribution is CIFAR-10, and the velocity field is represented by a small UNet.

The practical instructions are kept separate from the conceptual overview. For environment setup, dataset preparation, training, and sampling commands, see [Documentation](Documentation/README.md).

## The Idea

Diffusion models are often introduced through a noising process and a learned reverse process. Flow Matching takes a more direct view: start from noise, define a path toward data, and train a network to predict the velocity along that path.

The model defines an ODE:

```math
\frac{d x_t}{d t} = v_\theta(x_t, t).
```

If the learned velocity field is good, integrating this ODE from a noise sample produces an image:

```math
x_0 \sim \mathcal{N}(0, I),
\qquad
x_1 \approx \text{generated sample}.
```

So sampling is not framed as repeatedly denoising an image. It is framed as moving a point through a learned vector field.

## A Straight Path From Noise to Images

The cleanest version pairs a noise sample with a real image:

```math
z \sim \mathcal{N}(0, I),
\qquad
x \sim p_{\mathrm{data}}(x).
```

Then it draws a straight line between them:

```math
x_t = (1 - t)z + tx.
```

At the endpoints:

```math
x_{t=0} = z,
\qquad
x_{t=1} = x.
```

Taking the derivative with respect to time gives the velocity along this path:

```math
\frac{d x_t}{d t} = x - z.
```

That makes the supervised target unusually simple:

```math
u_t = x - z.
```

This path construction is implemented in [`flow/paths.py`](flow/paths.py).

## The Training Objective

During training, the code samples a real CIFAR-10 image, a Gaussian noise image, and a random time:

```math
x \sim p_{\mathrm{data}},
\qquad
z \sim \mathcal{N}(0, I),
\qquad
t \sim U(0, 1).
```

It constructs the interpolated point:

```math
x_t = (1 - t)z + tx,
```

and trains the network to predict the velocity from noise to data:

```math
\mathcal{L}_{FM}
=
\mathbb{E}_{x,z,t}
\left[
\left\|
v_\theta(x_t, t) - (x - z)
\right\|^2
\right].
```

The network does not receive the original pair $(x, z)$, only the current point $x_t$ and the time $t$. Under MSE training, the optimal prediction is therefore:

```math
v^*(x_t, t)
=
\mathbb{E}
\left[
x - z
\mid x_t, t
\right].
```

In other words, the model learns the average direction that points at a given location and time should move.

The loss wrapper lives in [`flow/losses.py`](flow/losses.py), and the training loop calls it from [`train.py`](train.py).

## Matching a Flow

The word "flow" refers to the motion of an entire probability distribution over time. Let $p_t(x)$ be the distribution of samples at time $t$, with:

```math
p_0(x) = \mathcal{N}(0, I),
\qquad
p_1(x) = p_{\mathrm{data}}(x).
```

A velocity field $v_t(x)$ moves this density according to the continuity equation:

```math
\frac{\partial p_t(x)}{\partial t}
+
\nabla \cdot \left(p_t(x)v_t(x)\right)
=
0.
```

This says that probability mass is not created or destroyed. It is transported by the velocity field. Flow Matching trains $v_\theta(x,t)$ to match that transport field.

## Sampling With the Learned Field

After training, generation starts from Gaussian noise:

```math
x_0 \sim \mathcal{N}(0, I).
```

The sampler integrates:

```math
\frac{d x_t}{d t} = v_\theta(x_t, t),
\qquad
t: 0 \rightarrow 1.
```

In this repository, sampling can use Euler or Heun integration. More steps usually give smoother trajectories, while fewer steps make generation faster.

The ODE integrator is implemented in [`flow/sampler.py`](flow/sampler.py), with the sampling CLI in [`sample.py`](sample.py).

## Relation to Diffusion

Diffusion models usually define a noising path such as:

```math
x_t = \alpha_t x_0 + \sigma_t \epsilon,
```

and train a model to predict noise, score, data, or a velocity-style parameterization derived from that noising process.

Flow Matching uses velocity more literally. For the straight path:

```math
x_t = (1 - t)z + tx,
\qquad
v = x - z.
```

The velocity is the derivative of the path itself.

| Method | Learned object | Generation |
| --- | --- | --- |
| VAE | latent distribution and decoder | sample latent, then decode |
| Diffusion | denoising, score, or noise field | reverse denoising or reverse SDE/ODE |
| Flow Matching | vector field / velocity field | solve an ODE from noise to data |

## Inside This Demo

The project trains a small UNet on normalized CIFAR-10 images:

```math
x \in \mathbb{R}^{3 \times 32 \times 32}.
```

The unconditional model learns:

```math
v_\theta(x_t, t).
```

The class-conditional model also receives a CIFAR-10 label $y$:

```math
v_\theta(x_t, t, y).
```

For conditional generation, the class embedding is added to the time embedding before it is passed through the UNet blocks. Sampling can then request a specific class, cycle through classes, or render a full class grid.

The UNet and class-conditioning logic live in [`models/unet.py`](models/unet.py).

The training loss used by the code is the same Flow Matching objective:

```math
\mathcal{L}
=
\mathbb{E}_{x,z,t}
\left[
\left\|
v_\theta(x_t,t,y) - (x - z)
\right\|^2
\right].
```

For unconditional training, the label term is simply absent.

## Results

Below are representative outputs from a CIFAR-10 class-conditional run.

### Class-Conditional Grid

Each row corresponds to one CIFAR-10 class. The class name is shown once on the left.

<img src="assets/results/class_grid_euler_050.png" alt="Class-conditional CIFAR-10 samples" width="520">

### Training Sample at Epoch 100

A sampling grid saved during training.

<img src="assets/results/epoch_0100.png" alt="CIFAR-10 samples at epoch 100" width="520">

## Takeaway

The whole demo is built around three equations:

```math
x_t = (1-t)z + tx,
\qquad
u_t = x - z,
```

```math
\mathcal{L}
=
\mathbb{E}_{x,z,t}
\left[
\left\|
v_\theta(x_t,t)-u_t
\right\|^2
\right],
```

```math
\frac{d x_t}{d t}=v_\theta(x_t,t).
```

Train the network to match the velocity on the path, then generate by following the learned velocity field from noise to image.
