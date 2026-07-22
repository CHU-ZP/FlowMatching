# CIFAR-10 上的 Flow Matching

<p align="right">
<a href="README.md">English</a> | 中文
</p>

这是一个用 CIFAR-10 演示 **pixel-space Flow Matching / Rectified Flow** 的小项目。

它做的事情很直接：

```text
Gaussian noise z (+ optional class label y) -> learned ODE flow -> CIFAR-10 image x
```

也就是从一张高斯噪声图像出发，沿着神经网络学到的 ODE 速度场流动，最后得到一张 32x32 的 RGB 图像。

这个 README 主要解释项目背后的想法，并展示一些生成结果。环境配置、数据准备、训练和采样命令放在 [Documentation](Documentation/README.md)。

## 直觉

扩散模型通常从“先加噪、再学习反向去噪”这个过程讲起。Flow Matching 的视角更像是：在噪声分布和真实数据分布之间铺一条路，然后学习这条路上的速度场。

模型学习的是一个连续时间 ODE：

```math
\frac{d x_t}{d t} = v_\theta(x_t, t).
```

如果速度场学得足够好，从高斯噪声开始沿着它积分，就能走到图像分布附近：

```math
x_0 \sim \mathcal{N}(0, I),
\qquad
x_1 \approx \text{generated sample}.
```

所以这里的采样不是反复“去噪”，而是在一个学到的向量场里移动。

## 一条从噪声到图像的直线

最简单的构造方式是：取一个噪声样本 $z$，再取一张真实图像 $x$：

```math
z \sim \mathcal{N}(0, I),
\qquad
x \sim p_{\mathrm{data}}(x).
```

然后用线性插值把它们连起来：

```math
x_t = (1 - t)z + tx.
```

当 $t=0$ 时，它就是噪声；当 $t=1$ 时，它就是真实图像：

```math
x_{t=0} = z,
\qquad
x_{t=1} = x.
```

这条路径对时间求导以后，速度非常简单：

```math
\frac{d x_t}{d t} = x - z.
```

也就是说，训练目标就是让网络预测：

```math
u_t = x - z.
```

这部分路径构造在 [`flow/paths.py`](flow/paths.py) 里。

## 训练到底在学什么

每次训练时，代码会采样三样东西：真实图像、噪声图像和一个随机时间点。

```math
x \sim p_{\mathrm{data}},
\qquad
z \sim \mathcal{N}(0, I),
\qquad
t \sim U(0, 1).
```

然后构造路径上的中间点：

```math
x_t = (1 - t)z + tx,
```

并让网络预测从噪声指向图像的速度：

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

这里有一个容易忽略的点：目标速度 $x-z$ 和当前随机配对的 $(x,z)$ 有关，所以它本身是随机的。网络看到的却只有 $x_t$ 和 $t$，并不知道原始的 $x$ 和 $z$。

在 MSE 训练下，最优预测会变成条件期望：

```math
v^*(x_t, t)
=
\mathbb{E}
\left[
x - z
\mid x_t, t
\right].
```

因此，模型不是在记住某一次随机配对的速度，而是在学习：处在这个位置、这个时间时，平均来说应该往哪个方向流动。

loss 的封装在 [`flow/losses.py`](flow/losses.py)，训练循环在 [`train.py`](train.py) 里调用它。

## 为什么叫 Flow

如果不只看单个样本，而是看整个分布，那么每个时间 $t$ 都有一个分布 $p_t(x)$：

```math
p_0(x) = \mathcal{N}(0, I),
\qquad
p_1(x) = p_{\mathrm{data}}(x).
```

速度场会推动这个分布随时间变化。这个过程满足连续性方程：

```math
\frac{\partial p_t(x)}{\partial t}
+
\nabla \cdot \left(p_t(x)v_t(x)\right)
=
0.
```

它表达的是一件很朴素的事：概率质量不会凭空出现或消失，只是在速度场的推动下移动。Flow Matching 要学的，就是这个推动分布流动的速度场。

## 怎么生成图像

训练完成后，采样时不再需要真实图像，只从高斯噪声开始：

```math
x_0 \sim \mathcal{N}(0, I).
```

然后求解：

```math
\frac{d x_t}{d t} = v_\theta(x_t, t),
\qquad
t: 0 \rightarrow 1.
```

这个项目支持 Euler 和 Heun 两种积分方式。步数更多时轨迹通常更平滑，步数更少时生成更快。

ODE sampler 在 [`flow/sampler.py`](flow/sampler.py)，采样入口在 [`sample.py`](sample.py)。

## 这个项目实现了什么

数据是归一化后的 CIFAR-10 图像：

```math
x \in \mathbb{R}^{3 \times 32 \times 32}.
```

无条件模型学习：

```math
v_\theta(x_t, t).
```

有条件模型额外接收 CIFAR-10 类别标签 $y$：

```math
v_\theta(x_t, t, y).
```

class-conditional 版本会把 class embedding 加到 time embedding 上，再送入 UNet 的各个 block。采样时可以指定某一类，也可以循环类别，或者生成每一行一个类别的 class grid。

UNet 和 class-conditioning 逻辑在 [`models/unet.py`](models/unet.py)。

代码里的训练目标仍然是同一个 Flow Matching loss：

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

无条件训练时，类别标签这一项会被省略。

## 和扩散模型的关系

扩散模型里的 velocity prediction 通常来自某个加噪公式，例如：

```math
x_t = \alpha_t x_0 + \sigma_t \epsilon,
```

然后把预测目标改写成 noise、score、data 或 velocity 的某种等价参数化。

Flow Matching 里的 velocity 更直接。对于这里的直线路径：

```math
x_t = (1 - t)z + tx,
\qquad
v = x - z.
```

这个 $v$ 就是路径本身的时间导数。

| 方法 | 学习对象 | 生成方式 |
| --- | --- | --- |
| VAE | latent distribution 和 decoder | sample latent 后 decode |
| Diffusion | denoising、score 或 noise field | 反向去噪或 reverse SDE/ODE |
| Flow Matching | vector field / velocity field | 从噪声出发解 ODE 到数据 |

## 结果展示

下面是一次 CIFAR-10 class-conditional 训练后的代表性结果。

### 按类别排列的条件生成

每一行对应一个 CIFAR-10 类别，类别名只在最左侧写一次。

<img src="assets/results/class_grid_euler_050.png" alt="CIFAR-10 类别条件生成样本" width="520">

### 第 100 个 Epoch 的训练采样

训练过程中保存的一张采样网格。

<img src="assets/results/epoch_0100.png" alt="第 100 个 epoch 的 CIFAR-10 采样结果" width="520">

## 总结

这个 demo 的核心就是三行公式：

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

训练时让网络匹配路径上的速度；生成时从噪声出发，沿着学到的速度场流到图像。
