# Documentation

这里放项目的实际使用方法。根目录 [README](../README.md) 主要介绍 Flow Matching 原理和 demo 结果展示。

## 1. 环境

进入项目目录：

```bash
cd /home/zepeng/DEMO/FlowMatching/cifar10_flow_matching
```

同步 uv 环境：

```bash
uv sync
```

项目默认使用 Python 3.12，并固定 PyTorch CUDA 12.8 wheel。检查：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

期望类似：

```text
2.11.0+cu128 12.8 True
```

## 2. 数据准备

默认数据源是 Hugging Face：

```yaml
data_source: huggingface
data_dir: ./datasets
hf_cache_dir: ./datasets/huggingface
hf_dataset_id: uoft-cs/cifar10
```

下载并校验 CIFAR-10：

```bash
uv run python prepare_data.py --config configs/cifar10_unet_fm.yaml
```

如果 Hugging Face 官方源慢，可以使用镜像：

```bash
uv run python prepare_data.py \
  --config configs/cifar10_unet_fm.yaml \
  --data-source huggingface \
  --hf-endpoint https://hf-mirror.com
```

也可以在完整流程脚本中指定：

```bash
scripts/full_pipeline.sh \
  --mode conditional \
  --epochs 100 \
  --data-source huggingface \
  --hf-endpoint https://hf-mirror.com
```

如果要回退到 torchvision 数据源：

```bash
uv run python prepare_data.py \
  --config configs/cifar10_unet_fm.yaml \
  --data-source torchvision
```

并把配置改成：

```yaml
data_source: torchvision
data_dir: ./datasets
```

## 3. 训练

默认训练有条件版本：

```bash
uv run python train.py --config configs/cifar10_unet_fm.yaml
```

输出目录：

```text
runs/cifar10_fm/
├── checkpoints/latest.pt
└── samples/
```

训练过程中保存的预览图片会写到：

```text
runs/cifar10_fm/samples/
```

中断后恢复：

```bash
uv run python train.py \
  --config configs/cifar10_unet_fm.yaml \
  --resume runs/cifar10_fm/checkpoints/latest.pt
```

本地显存较小时，建议先把配置里的 batch size 改小：

```yaml
batch_size: 64
```

或：

```yaml
batch_size: 32
```

## 4. 采样

默认 class cycle 采样：

```bash
uv run python sample.py \
  --config configs/cifar10_unet_fm.yaml \
  --ckpt runs/cifar10_fm/checkpoints/latest.pt \
  --num-samples 64 \
  --steps 50
```

每个类别一行：

```bash
uv run python sample.py \
  --ckpt runs/cifar10_fm/checkpoints/latest.pt \
  --class-grid \
  --samples-per-class 8 \
  --steps 50
```

`--class-grid` 会在每行左侧写一次类别名。

指定类别，例如 `3 = cat`：

```bash
uv run python sample.py \
  --ckpt runs/cifar10_fm/checkpoints/latest.pt \
  --class-label 3 \
  --num-samples 64 \
  --steps 50
```

关闭类别文字：

```bash
uv run python sample.py \
  --ckpt runs/cifar10_fm/checkpoints/latest.pt \
  --class-grid \
  --samples-per-class 8 \
  --steps 50 \
  --no-class-names
```

保存生成过程 GIF：

```bash
uv run python sample.py \
  --ckpt runs/cifar10_fm/checkpoints/latest.pt \
  --class-grid \
  --samples-per-class 8 \
  --steps 50 \
  --save-trajectory \
  --trajectory-every 2
```

比较不同 ODE 步数：

```bash
uv run python sample.py --ckpt runs/cifar10_fm/checkpoints/latest.pt --steps 10
uv run python sample.py --ckpt runs/cifar10_fm/checkpoints/latest.pt --steps 20
uv run python sample.py --ckpt runs/cifar10_fm/checkpoints/latest.pt --steps 50
uv run python sample.py --ckpt runs/cifar10_fm/checkpoints/latest.pt --steps 100
```

## 5. 无条件版本

把配置改成：

```yaml
class_conditional: false
```

然后重新训练：

```bash
uv run python train.py --config configs/cifar10_unet_fm.yaml
```

无条件采样：

```bash
uv run python sample.py \
  --ckpt runs/cifar10_fm/checkpoints/latest.pt \
  --num-samples 64 \
  --steps 50
```

## 6. 一键完整流程

条件版本：

```bash
scripts/full_pipeline.sh \
  --mode conditional \
  --epochs 100 \
  --batch-size 64 \
  --data-source huggingface \
  --hf-endpoint https://hf-mirror.com
```

无条件版本：

```bash
scripts/full_pipeline.sh \
  --mode unconditional \
  --epochs 100 \
  --batch-size 64 \
  --data-source huggingface \
  --hf-endpoint https://hf-mirror.com
```

两套都跑：

```bash
scripts/full_pipeline.sh \
  --mode both \
  --epochs 100 \
  --batch-size 64 \
  --data-source huggingface \
  --hf-endpoint https://hf-mirror.com
```

查看完整参数：

```bash
scripts/full_pipeline.sh --help
```

完整流程脚本会把生成结果写到：

```text
runs/full_pipeline/
├── conditional/samples/
└── unconditional/samples/
```

## 7. CIFAR-10 类别

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
