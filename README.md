# CIFAR-10 Flow Matching Demo

这是一个最小、直观的 **pixel-space class-conditional Flow Matching** demo：

```text
Gaussian noise x0 + class label y -> ODE flow -> CIFAR-10 image x1
```

训练目标是让 UNet 在随机时间 `t` 的线性路径上，结合类别标签 `y` 预测速度：

```text
xt = (1 - t) * x0 + t * x1
v_theta(xt, t, y) ~= x1 - x0
```

## 项目结构

```text
cifar10_flow_matching/
├── configs/cifar10_unet_fm.yaml
├── flow/
│   ├── paths.py
│   ├── sampler.py
│   └── losses.py
├── models/unet.py
├── data/cifar10.py
├── pyproject.toml
├── train.py
└── sample.py
```

## 使用 uv 安装环境

```bash
cd cifar10_flow_matching
uv sync
```

项目默认通过 `.python-version` 使用 Python 3.12。`uv sync` 会自动创建 `.venv/` 并安装 `pyproject.toml` 里的依赖。

PyTorch 依赖固定从官方 `cu128` index 解析，也就是 CUDA 12.8 wheel。同步后可以检查：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

输出应包含类似 `+cu128` 和 `12.8`。

## 训练

```bash
uv run python train.py --config configs/cifar10_unet_fm.yaml
```

CIFAR-10 会自动下载到 `datasets/`。checkpoint 和采样图会保存到：

```text
runs/cifar10_fm/
├── checkpoints/latest.pt
└── samples/
```

## 采样

```bash
uv run python sample.py \
  --config configs/cifar10_unet_fm.yaml \
  --ckpt runs/cifar10_fm/checkpoints/latest.pt \
  --num-samples 64 \
  --steps 50 \
  --method euler
```

指定单个 CIFAR-10 类别生成，例如 `3 = cat`：

```bash
uv run python sample.py \
  --ckpt runs/cifar10_fm/checkpoints/latest.pt \
  --class-label 3 \
  --num-samples 64 \
  --steps 50
```

生成每个类别一行的网格：

```bash
uv run python sample.py \
  --ckpt runs/cifar10_fm/checkpoints/latest.pt \
  --class-grid \
  --samples-per-class 8 \
  --steps 50
```

保存从 `t=0` 到 `t=1` 的生成过程：

```bash
uv run python sample.py \
  --ckpt runs/cifar10_fm/checkpoints/latest.pt \
  --num-samples 64 \
  --steps 50 \
  --save-trajectory \
  --trajectory-every 2
```

也可以比较不同 ODE 步数：

```bash
uv run python sample.py --steps 10
uv run python sample.py --steps 20
uv run python sample.py --steps 50
uv run python sample.py --steps 100
```

## 常用 uv 命令

```bash
uv sync                 # 创建/同步环境
uv run python train.py  # 在 uv 环境中运行命令
uv add package-name     # 添加依赖
uv lock                 # 更新 uv.lock
```

## 配置

默认配置在 `configs/cifar10_unet_fm.yaml`：

```yaml
batch_size: 128
lr: 0.0002
epochs: 100
base_channels: 64
channel_mult: [1, 2, 2]
num_res_blocks: 2
time_embedding_dim: 256
class_conditional: true
num_classes: 10
ema: true
num_steps_sampling: 50
```

CIFAR-10 label 对应：

```text
0 airplane
1 automobile
2 bird
3 cat
4 deer
5 dog
6 frog
7 horse
8 ship
9 truck
```

如果想回到无条件版本，把配置里的 `class_conditional` 改成 `false` 即可。

第一个目标不是 FID，而是看见 loss 稳定下降，并且从噪声采样逐渐出现颜色块、纹理和物体轮廓。
