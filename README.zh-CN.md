# Modular Flow Matching

<p align="right">
<a href="README.md">English</a> | 中文
</p>

这份仓库用一套尽量精简的 PyTorch 实现来解释 Flow Matching，其中包括 Rectified Flow、类别条件速度预测和 ODE 采样。

```text
高斯噪声（+可选类别标签）-> 学到的速度场 -> CIFAR10 图像
```

概率路径、训练目标、速度网络和数值求解器都被单独拆开，因此既容易顺着公式读代码，也方便替换其中任意一部分。

环境、数据准备、训练和采样命令见 [Documentation](Documentation/README.md)。

## 基本想法

Flow Matching 要学习的是一个随时间变化的速度场，它负责把简单分布逐渐搬运到数据分布：

```math
\frac{d x_t}{d t} = v_\theta(x_t, t).
```

生成时从标准高斯噪声出发，沿着学到的速度场求解 ODE，从 `t=0` 一直走到 `t=1`：

```math
x_0 \sim \mathcal{N}(0, I),
\qquad
x_1 \sim p_{\mathrm{data}}.
```

网络不需要预测某个离散的反向转移，只要回答“当前位置接下来该往哪走”。具体怎样沿速度场前进，则交给数值求解器。

如果不再盯着单个样本，而是看整个分布，这条路径连接的是高斯分布和真实数据分布：

```math
p_0(x)=\mathcal{N}(0,I),
\qquad
p_1(x)=p_{\mathrm{data}}(x).
```

随时间变化的密度和速度场满足连续性方程：

```math
\frac{\partial p_t(x)}{\partial t}
+
\nabla\cdot\left(p_t(x)v_t(x)\right)
=0.
```

连续性方程表达的是：概率质量不会凭空出现或消失，只会在速度场的推动下移动。Flow Matching 学的正是推动这股概率流的方向和速度。

## 线性概率路径

最简单的做法是分别取一个噪声样本 `z` 和一个真实样本 `x`，再用线性插值把二者连起来：

```math
z \sim \mathcal{N}(0, I),
\qquad
x \sim p_{\mathrm{data}}.
```

```math
x_t = (1-t)z + tx.
```

在 `t=0` 时路径位于噪声 `z`，在 `t=1` 时到达数据 `x`。对这条路径求时间导数，便得到训练时使用的目标速度：

```math
u_t = \frac{d x_t}{d t} = x-z.
```

这就是最简单的 Rectified Flow。因为路径是一条直线，所以给定这一对端点后，速度始终是 `x_1-x_0`。

路径构造实现在 [`flow/paths.py`](flow/paths.py)。

## 训练目标

每一步训练都会取一张真实图像、一份高斯噪声和一个均匀采样的时间。网络看到插值点 `x_t` 后，要让预测速度尽量接近这条路径的真实速度：

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

这里有一点容易混淆：目标 `x-z` 取决于这一次随机配对，而网络只能看到 `(x_t,t)`，并不知道原来的 `x` 和 `z`。在均方误差下，网络最终学到的是给定当前位置和时间后的平均速度，也就是下面的条件期望：

```math
v^*(x_t,t)
=
\mathbb{E}
\left[
x-z \mid x_t,t
\right].
```

损失函数封装在 [`flow/losses.py`](flow/losses.py)，训练循环位于 [`train.py`](train.py)。

## 采样

采样时只需要一份高斯噪声，然后对网络给出的速度场做数值积分：

```math
\frac{d x_t}{d t}=v_\theta(x_t,t),
\qquad
t:0\rightarrow1.
```

代码提供 Euler 和 Heun 两种积分方法。Euler 每一步只计算一次速度；Heun 会多做一次预测—校正，轨迹通常更准确。增加步数往往能让轨迹更平滑，减少步数则能换来更快的生成速度。

ODE 采样器实现在 [`flow/sampler.py`](flow/sampler.py)。

## 代码与实验

实验直接在归一化后的 CIFAR10 像素空间中进行：

```math
x \in \mathbb{R}^{3\times32\times32}.
```

同一个 UNet 既可以做无条件训练，也可以额外接收 CIFAR10 类别标签：

```math
v_\theta(x_t,t),
\qquad
v_\theta(x_t,t,y).
```

做类别条件训练时，类别嵌入会与时间嵌入相加，再送入 UNet 的各个残差块。采样时既可以指定某一类，也可以循环类别，或者生成每行对应一个类别的网格。

| 组件 | 实现 |
| --- | --- |
| 线性概率路径 | [`flow/paths.py`](flow/paths.py) |
| Flow Matching 损失 | [`flow/losses.py`](flow/losses.py) |
| Euler 和 Heun 采样器 | [`flow/sampler.py`](flow/sampler.py) |
| 条件 UNet | [`models/unet.py`](models/unet.py) |
| 训练入口 | [`train.py`](train.py) |
| 采样入口 | [`sample.py`](sample.py) |

## 和其他生成方法放在一起看

| 方法 | 学习对象 | 生成方式 |
| --- | --- | --- |
| VAE | 近似后验、先验和解码器 | 采样潜变量后解码 |
| Diffusion | 噪声、score、数据或速度 | 反向 SDE、ODE 或离散去噪链 |
| Flow Matching | 连续速度场 | 从噪声出发解 ODE |
| 这份实现 | 像素空间中的类别条件速度场 | 在 CIFAR10 上用 Euler 或 Heun 积分 |

扩散模型里的速度预测通常来自预先规定的加噪路径：

```math
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad
v=\alpha_t\epsilon-\sigma_t x_0.
```

在扩散模型中，速度只是预测目标的一种参数化方式。这里的 Flow Matching 不同：对于这条直线路径，速度就是路径本身对时间的导数。

```math
x_t=(1-t)z+tx,
\qquad
v=x-z.
```

## 结果

下面展示一组已经完成的 CIFAR10 类别条件实验。原始结果文件保存在 [`assets/results/`](assets/results) 中。

训练脚本和完整流程会把生成样本写入：

```text
runs/cifar10_fm/samples/
runs/full_pipeline/conditional/samples/
runs/full_pipeline/unconditional/samples/
```

### 按类别排列的生成结果

每一行对应一个 CIFAR10 类别。

<img src="assets/results/class_grid_euler_050.png" alt="CIFAR10 Flow Matching 类别条件生成样本" width="520">

### 第 100 轮训练时的采样

<img src="assets/results/epoch_0100.png" alt="Flow Matching 第 100 个 epoch 的 CIFAR10 采样" width="520">

常见的生成结果包括：

| 结果 | 用途 | 默认文件名 |
| --- | --- | --- |
| 类别网格 | 每行一个 CIFAR10 类别 | `class_grid_euler_050.png` |
| 类别循环 | 快速检查类别控制是否生效 | `class_cycle_euler_050.png` |
| 单一类别 | 只生成指定类别，例如 `cat` | `class_3_euler_050.png` |
| 无条件采样 | 无条件模型的生成结果 | `samples_euler_050.png` |
| 生成轨迹 | 从噪声到图像的 ODE 过程 | `*.gif` |

## 小结

理解 Flow Matching 可以抓住三件事：概率路径先规定噪声与数据怎样连接，网络再学习路径上的速度，最后由 ODE 求解器沿着速度场生成样本。

VAE 从潜空间采样后直接解码；扩散模型学习反向去噪；Flow Matching 则学习一片能够把噪声连续搬运到数据分布的速度场。

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
