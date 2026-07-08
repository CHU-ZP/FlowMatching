# CIFAR-10 上的 Flow Matching

<p align="right">
<a href="README.md">English</a> | 中文
</p>

这个仓库是一个小而直接的 **CIFAR-10 pixel-space Flow Matching / Rectified Flow** 展示项目。

```text
Gaussian noise z (+ optional class label y) -> learned ODE flow -> CIFAR-10 image x
```

Flow Matching 可以用一句话理解：

> 学习一个连续时间速度场，把简单噪声分布搬运到数据分布。

在这个项目里，简单分布是标准高斯分布，数据分布是 CIFAR-10，速度场由一个 small UNet 表示。

具体运行方法和原理介绍分开放置。环境配置、数据准备、训练和采样命令见 [Documentation](Documentation/README.md)。

## 核心直觉

Diffusion model 通常从一个加噪过程和一个学习到的反向过程开始理解。Flow Matching 的视角更直接：从噪声出发，定义一条通向数据的路径，然后训练网络预测这条路径上的速度。

模型定义一个 ODE：

```math
\frac{d x_t}{d t} = v_\theta(x_t, t).
```

如果学习到的速度场足够好，从噪声样本出发积分这个 ODE，就可以得到图像：

```math
x_0 \sim \mathcal{N}(0, I),
\qquad
x_1 \approx \text{generated sample}.
```

因此，采样不是反复去噪，而是在学习到的向量场中移动一个点。

## 从噪声到图像的直线路径

最干净的版本会把一个噪声样本和一张真实图像配对：

```math
z \sim \mathcal{N}(0, I),
\qquad
x \sim p_{\mathrm{data}}(x).
```

然后在二者之间画一条直线：

```math
x_t = (1 - t)z + tx.
```

在两个端点处：

```math
x_{t=0} = z,
\qquad
x_{t=1} = x.
```

对时间求导，就得到这条路径上的速度：

```math
\frac{d x_t}{d t} = x - z.
```

所以监督目标非常简单：

```math
u_t = x - z.
```

这部分路径构造实现在 [`flow/paths.py`](flow/paths.py)。

## 训练目标

训练时，代码会采样一张真实 CIFAR-10 图像、一张高斯噪声图像，以及一个随机时间：

```math
x \sim p_{\mathrm{data}},
\qquad
z \sim \mathcal{N}(0, I),
\qquad
t \sim U(0, 1).
```

然后构造插值点：

```math
x_t = (1 - t)z + tx,
```

并训练网络预测从噪声指向数据的速度：

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

网络并不会看到原始的 $(x, z)$，只会看到当前位置 $x_t$ 和时间 $t$。在 MSE 训练下，最优预测是条件期望：

```math
v^*(x_t, t)
=
\mathbb{E}
\left[
x - z
\mid x_t, t
\right].
```

也就是说，模型学到的是在某个位置和时间下，样本平均应该往哪个方向移动。

loss 封装在 [`flow/losses.py`](flow/losses.py)，训练循环在 [`train.py`](train.py) 中调用它。

## 匹配一个概率流

"flow" 指的是整个概率分布随时间的运动。令 $p_t(x)$ 表示时间 $t$ 的样本分布，并满足：

```math
p_0(x) = \mathcal{N}(0, I),
\qquad
p_1(x) = p_{\mathrm{data}}(x).
```

速度场 $v_t(x)$ 会按照连续性方程推动这个密度：

```math
\frac{\partial p_t(x)}{\partial t}
+
\nabla \cdot \left(p_t(x)v_t(x)\right)
=
0.
```

这个方程表达的是：概率质量不会凭空出现或消失，只会被速度场搬运。Flow Matching 训练 $v_\theta(x,t)$ 去匹配这个搬运场。

## 使用学习到的速度场采样

训练完成后，生成从高斯噪声开始：

```math
x_0 \sim \mathcal{N}(0, I).
```

采样器积分：

```math
\frac{d x_t}{d t} = v_\theta(x_t, t),
\qquad
t: 0 \rightarrow 1.
```

这个仓库支持 Euler 和 Heun 积分。更多步数通常会得到更平滑的轨迹，更少步数则生成更快。

ODE 积分器实现在 [`flow/sampler.py`](flow/sampler.py)，采样命令入口在 [`sample.py`](sample.py)。

## 和 Diffusion 的关系

Diffusion model 通常定义类似下面的加噪路径：

```math
x_t = \alpha_t x_0 + \sigma_t \epsilon,
```

然后训练模型预测 noise、score、data，或者从这个加噪过程推导出的 velocity 参数化。

Flow Matching 里的 velocity 更直接。对直线路径来说：

```math
x_t = (1 - t)z + tx,
\qquad
v = x - z.
```

这个 velocity 就是路径本身的时间导数。

| 方法 | 学习对象 | 生成方式 |
| --- | --- | --- |
| VAE | latent distribution 和 decoder | sample latent 后 decode |
| Diffusion | denoising、score 或 noise field | 反向去噪或 reverse SDE/ODE |
| Flow Matching | vector field / velocity field | 从噪声出发解 ODE 到数据 |

## 这个 Demo 里实现了什么

项目在归一化后的 CIFAR-10 图像上训练 small UNet：

```math
x \in \mathbb{R}^{3 \times 32 \times 32}.
```

无条件模型学习：

```math
v_\theta(x_t, t).
```

有条件模型还会接收 CIFAR-10 类别标签 $y$：

```math
v_\theta(x_t, t, y).
```

在有条件生成中，class embedding 会加到 time embedding 上，再送入 UNet blocks。采样时可以指定某一类、循环类别，或生成完整的 class grid。

UNet 和 class-conditioning 逻辑实现在 [`models/unet.py`](models/unet.py)。

代码中的训练 loss 仍然是同一个 Flow Matching 目标：

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

无条件训练时，类别项会被省略。

## 结果展示

下面是一次 CIFAR-10 类别条件生成实验的代表性输出。

### 按类别排列的条件生成结果

每一行对应一个 CIFAR-10 类别，类别名只在左侧写一次。

<img src="assets/results/class_grid_euler_050.png" alt="CIFAR-10 类别条件生成样本" width="520">

### 第 100 个 Epoch 的训练采样

训练过程中保存的一张采样网格。

<img src="assets/results/epoch_0100.png" alt="第 100 个 epoch 的 CIFAR-10 采样结果" width="520">

## 总结

整个 demo 可以浓缩成三个公式：

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

训练时让网络匹配路径上的速度；生成时从噪声出发，沿着学习到的速度场流向图像。
