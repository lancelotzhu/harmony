# AI Model Training Workspace

这是一个用于组织和开展不同 AI 模型训练实验的工作区。各训练方向以独立子项目的形式放在 `projects/` 下，分别维护自己的代码、配置、数据、模型权重、日志和运行说明，便于在同一仓库中扩展新的模型与训练流程。

## 子项目

| 子项目 | 说明 |
| --- | --- |
| [voice-cloning](projects/voice-cloning/README.md) | 基于 GPT-SoVITS 的声音复刻、语音合成与模型训练 |
| [qwen-mlx-finetune](projects/qwen-mlx-finetune/README.md) | 在 Apple Silicon 上对 Qwen3.5-0.8B 做 LoRA 微调，导出可供 oMLX 导入的量化 MLX 模型 |

## 开始使用

进入目标子项目并按照其中的说明准备环境、数据和预训练权重。例如，声音复刻项目的命令需在其目录中运行：

```bash
cd projects/voice-cloning
bash setup_env.sh
python cli.py --help
```

## 项目结构约定

```text
projects/
  <project-name>/
    README.md       # 项目介绍、环境和操作说明
    config.*        # 项目配置
    data/           # 数据集与预处理结果
    models/         # 预训练和训练产物
    logs/           # 训练日志
    output/         # 推理或评估结果
```

新增训练方向时，在 `projects/` 下建立独立目录，并将该方向专用的代码和资源放在其中。数据集、检查点和实验输出通常较大，仓库忽略规则默认不跟踪这些运行产物。
