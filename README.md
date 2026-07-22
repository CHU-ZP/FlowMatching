# CIFAR-10 Flow Matching Demo

这个项目展示一个最小、直观的 **CIFAR-10 pixel-space Flow Matching / Rectified Flow**：

```text
Gaussian noise z + optional class label y -> ODE flow -> CIFAR-10 image x
```

Flow Matching 可以理解为：

> 不是学习“如何一步步加噪再反向去噪”，而是直接学习一个连续时间速度场，把简单分布搬运到数据分布。

也就是从：

$$
z \sim p_0(z)
$$

通常是标准高斯：

$$
p_0 = \mathcal{N}(0,I)
$$

通过一个 ODE 流：

$$
\frac{dx_t}{dt} = v_\theta(x_t,t)
$$

把它变成真实数据分布：

$$
x_1 \sim p_{\text{data}}(x)
$$

具体运行、环境、数据准备、训练和采样方法放在 [Documentation](Documentation/README.md)。

## 1. Flow Matching 的核心对象

Diffusion 里模型常学的是：

$$
\epsilon_\theta(x_t,t)
$$

或者：

$$
x_{0,\theta}(x_t,t)
$$

或者：

$$
v_\theta(x_t,t)
$$

而 Flow Matching 里，模型直接学的是一个**速度场**：

$$
v_\theta(x_t,t)
$$

它回答的问题是：

> 当前点在时间 $t$ 位于 $x_t$，它应该往哪个方向移动，才能最终变成数据样本？

所以 sampling 时不是反复去噪，而是解一个 ODE：

$$
x_0 \sim \mathcal{N}(0,I)
$$

$$
\frac{dx_t}{dt}=v_\theta(x_t,t)
$$

$$
x_1 \approx \text{generated sample}
$$

## 2. 最简单的 Flow Matching：线性插值路径

取一个噪声样本：

$$
z \sim \mathcal{N}(0,I)
$$

取一个真实数据样本：

$$
x \sim p_{\text{data}}(x)
$$

构造从噪声到数据的路径：

$$
x_t = (1-t)z + tx
$$

当：

$$
t=0
$$

有：

$$
x_0=z
$$

当：

$$
t=1
$$

有：

$$
x_1=x
$$

这是一条从噪声点走到真实样本点的直线路径。

对时间求导：

$$
\frac{dx_t}{dt}=x-z
$$

所以目标速度就是：

$$
u_t = x-z
$$

训练目标变成：

$$
\mathcal{L}_{FM}
=
\mathbb{E}_{x,z,t}
\left[
\left\|
v_\theta(x_t,t) - (x-z)
\right\|^2
\right]
$$

这就是最直观版本的 Flow Matching。

## 3. 为什么叫 Flow Matching？

因为它要匹配一个概率流。

假设每个时间都有一个分布：

$$
p_t(x)
$$

其中：

$$
p_0(x)=\mathcal{N}(0,I)
$$

$$
p_1(x)=p_{\text{data}}(x)
$$

如果有一个速度场：

$$
v_t(x)
$$

那么分布会按照这个速度场流动。

这个过程满足连续性方程：

$$
\frac{\partial p_t(x)}{\partial t}
+
\nabla \cdot
\left(
p_t(x)v_t(x)
\right)
=0
$$

这个式子表达的是：

> 概率质量不会凭空出现或消失，只是在速度场推动下移动。

Flow Matching 的目标就是学习一个神经网络速度场：

$$
v_\theta(x,t)
$$

让它匹配真实的概率流速度：

$$
v_t(x)
$$

## 4. 一个关键问题：速度目标不是随机的吗？

训练时我们采样：

$$
z \sim \mathcal{N}(0,I)
$$

$$
x \sim p_{\text{data}}(x)
$$

$$
t \sim U(0,1)
$$

然后构造：

$$
x_t=(1-t)z+tx
$$

目标速度是：

$$
x-z
$$

这当然和采样的 $(x,z)$ 有关，看起来也是随机的。

但这和 diffusion 里预测噪声很类似。模型看到的是：

$$
x_t,t
$$

它并不知道原始的 $(x,z)$。在 MSE 训练下，最优预测会变成条件期望：

$$
v^*(x_t,t)
=
\mathbb{E}
\left[
x-z
\mid x_t,t
\right]
$$

也就是说，模型不是要记住某一次随机采样的速度，而是学习：

> 在当前位置 $x_t$ 和时间 $t$ 下，平均来说应该往哪里流动。

这点非常重要。

## 5. Rectified Flow 是最容易理解的特殊情况

Rectified Flow 常用的形式就是：

$$
x_t = (1-t)x_0 + tx_1
$$

其中：

$$
x_0 \sim \mathcal{N}(0,I)
$$

$$
x_1 \sim p_{\text{data}}
$$

目标速度：

$$
\dot{x}_t = x_1 - x_0
$$

训练：

$$
\mathbb{E}
\left[
\left\|
v_\theta(x_t,t)
-
(x_1-x_0)
\right\|^2
\right]
$$

采样：

$$
x_0 \sim \mathcal{N}(0,I)
$$

然后解：

$$
\frac{dx_t}{dt}
=
v_\theta(x_t,t)
$$

从 $t=0$ 积分到 $t=1$。

这就是最基础的 Flow Matching / Rectified Flow 直觉。

## 6. 本项目对应的实现

这个 demo 使用 CIFAR-10 图片作为数据样本：

$$
x \in \mathbb{R}^{3 \times 32 \times 32}
$$

噪声来自：

$$
z \sim \mathcal{N}(0,I)
$$

路径是：

$$
x_t = (1-t)z + tx
$$

网络是一个 small UNet。无条件版本学习：

$$
v_\theta(x_t,t)
$$

有条件版本额外输入 CIFAR-10 类别 $y$，学习：

$$
v_\theta(x_t,t,y)
$$

训练目标是：

$$
\mathcal{L}
=
\mathbb{E}_{x,z,t}
\left[
\left\|
v_\theta(x_t,t,y)-(x-z)
\right\|^2
\right]
$$

采样时从纯噪声开始，使用 Euler 或 Heun 方法积分：

$$
z \sim \mathcal{N}(0,I)
$$

$$
\frac{dx_t}{dt}=v_\theta(x_t,t,y)
$$

$$
x_1 = \text{generated sample}
$$

## 7. 它和 Diffusion 的区别

Diffusion 的思路是：

$$
x_0 \rightarrow x_t \rightarrow x_T
$$

先定义一个加噪过程：

$$
q(x_t \mid x_0)
$$

然后学习反向过程：

$$
p_\theta(x_{t-1} \mid x_t)
$$

或者连续时间下学习 score / noise / velocity。

Flow Matching 的思路是：

$$
z \rightarrow x
$$

直接定义一条从噪声分布到数据分布的路径，然后学习路径上的速度场。

| 方法 | 学习对象 | 生成方式 |
| --- | --- | --- |
| VAE | latent posterior / decoder | sample latent 后 decoder |
| Diffusion | denoising / score / noise | 反向去噪或 reverse SDE/ODE |
| Flow Matching | vector field / velocity field | 解 ODE 从噪声流到数据 |

Flow Matching 更像是在学：

> 如何把整个噪声分布连续地推送成数据分布。

## 8. Flow Matching 和 velocity prediction 的关系

Diffusion 里的 velocity 通常来自加噪公式：

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon
$$

对应的 velocity target 常写成某种组合：

$$
v = \alpha_t \epsilon - \sigma_t x_0
$$

它本质上还是服务于 diffusion 的噪声路径。

而 Flow Matching 里的 velocity 更直接：

$$
x_t = (1-t)z + tx
$$

$$
v = x-z
$$

它就是路径的时间导数。

所以可以粗略理解为：

> Diffusion 的 velocity prediction 是在 diffusion 框架内部重新参数化预测目标；Flow Matching 是直接把 velocity field 当成核心建模对象。

## 9. 结果展示

训练脚本会定期把采样结果保存到：

```text
runs/cifar10_fm/samples/
```

一键脚本会把结果保存到：

```text
runs/full_pipeline/
├── conditional/samples/
└── unconditional/samples/
```

下面是一次 CIFAR-10 class-conditional 训练后的示例结果。

**Class-Conditional Grid**

每一行对应一个 CIFAR-10 类别，左侧是类别名。

![Class-conditional CIFAR-10 samples](assets/results/class_grid_euler_050.png)

**Training Sample at Epoch 100**

训练过程中定期保存的采样网格。

![CIFAR-10 samples at epoch 100](assets/results/epoch_0100.png)

常见结果文件：

| 结果 | 说明 | 默认文件 |
| --- | --- | --- |
| class grid | 每行一个 CIFAR-10 类别，左侧写类别名 | `class_grid_euler_050.png` |
| class cycle | 类别循环采样，用于快速看条件控制是否生效 | `class_cycle_euler_050.png` |
| single class | 指定单个类别，例如 `cat` | `class_3_euler_050.png` |
| unconditional | 无条件版本的生成结果 | `samples_euler_050.png` |
| trajectory | 从噪声到图像的 ODE 生成过程 | `*.gif` |

## 10. 一句话总结

VAE 是：

> 学一个 latent distribution，再 decode。

Diffusion 是：

> 学一个反向去噪过程，把噪声逐步还原成数据。

Flow Matching 是：

> 学一个连续速度场，把噪声分布直接流动成数据分布。

最核心公式可以记成：

$$
x_t=(1-t)z+tx
$$

$$
u_t=x-z
$$

$$
\mathcal{L}
=
\mathbb{E}_{x,z,t}
\left[
\left\|
v_\theta(x_t,t)-u_t
\right\|^2
\right]
$$

生成时：

$$
z \sim \mathcal{N}(0,I)
$$

$$
\frac{dx_t}{dt}=v_\theta(x_t,t)
$$

$$
x_1 = \text{generated sample}
$$
