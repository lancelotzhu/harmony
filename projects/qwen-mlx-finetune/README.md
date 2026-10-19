# Qwen3.5-0.8B LoRA 微调与 MLX 导出

在 Apple Silicon macOS 上用 [MLX-LM](https://github.com/ml-explore/mlx-lm) 微调 Qwen3.5-0.8B，将 LoRA adapter 合并回模型并量化为 MLX 模型目录，再导入 oMLX 使用。

此项目只提供训练流程和数据格式检查；训练数据由你自行放入 `data/incoming/`，不会自动读取 `~/.config/opencode` 或复制任何个人数据。

## 环境

建议使用 Apple Silicon Mac 和 Python 3.11 的独立虚拟环境：

```bash
cd projects/qwen-mlx-finetune
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`mlx-lm[train]` 提供 MLX-LM 训练、合并和转换命令。确认当前版本支持目标模型：

```bash
python -m mlx_lm lora --help
python -m mlx_lm fuse --help
python -m mlx_lm convert --help
```

MLX 仅适用于 Apple Silicon。Qwen3.5 是较新的模型架构；若你安装的 MLX-LM 版本尚不支持 `Qwen/Qwen3.5-0.8B-Base`，训练或转换命令会报出架构错误。此时先升级 `mlx-lm[train]`，并检查该版本的模型支持列表，不要将失败训练产生的 adapter 当作有效结果。

## 准备数据

把你自行导出的 JSONL 文件放入 `data/incoming/`，例如 `data/incoming/train.jsonl`。格式采用 OpenCode 数据导出使用的 ShareGPT `messages` 形状，每行一个独立样本：

```jsonl
{"messages":[{"role":"user","content":"输入内容"},{"role":"assistant","content":"期望的回答"}]}
{"messages":[{"role":"system","content":"任务说明"},{"role":"user","content":"另一个输入"},{"role":"assistant","content":"另一个期望回答"}]}
```

如果使用 OpenCode 权限数据集，可在 OpenCode 配置目录运行已有导出脚本，再将其输出复制到本项目：

```bash
cd "$HOME/.config/opencode"
npx tsx dataset/export.ts
cp "$HOME/.local/share/opencode-permission/train.jsonl" \
  /path/to/harmony/projects/qwen-mlx-finetune/data/incoming/opencode-train.jsonl
```

导出脚本会更新 `~/.local/share/opencode-permission/` 下的训练集、元数据和报告；训练副本及模型产物保留在本机且被 Git 忽略。

支持 `system`、`user`、`assistant` 角色和字符串形式的 `content`。样本必须至少包含一条 user 消息和一条 assistant 消息，且最后一条消息必须是 assistant。每行必须是合法 JSON；不支持 tool call 或多模态消息。

校验输入并按固定随机种子拆分训练集和验证集：

```bash
python prepare_data.py \
  --input data/incoming/train.jsonl \
  --output-dir data/processed \
  --validation-fraction 0.1 \
  --seed 42
```

脚本生成 `data/processed/train.jsonl` 和（样本数足够时）`data/processed/valid.jsonl`。样本少于两条时会只生成训练集。验证集比例须在 `[0, 1)` 范围内。数据只会被校验、拆分，不会重写消息内容；数据目录已加入 Git 忽略规则。

划分按样本随机进行，不做类别分层。若少数类别样本很少，训练/验证集可能失去该类别；训练前应检查各类别数量，确保验证集包含需要评估的类别。

## LoRA 训练

默认模型为 Hugging Face 上的 `Qwen/Qwen3.5-0.8B-Base`。先完成上面的数据准备，再运行：

```bash
bash train.sh
```

脚本默认设置为 1000 iterations、batch size 1、训练最后 2 层、learning rate `1e-5`，并启用 prompt masking（只对 assistant 回答计算 loss）。这是在 24 GB Apple Silicon Mac 上验证可运行的低内存配置；可通过环境变量调整，例如：

```bash
MODEL="Qwen/Qwen3.5-0.8B-Base" \
ITERS=1500 BATCH_SIZE=1 NUM_LAYERS=8 LEARNING_RATE=1e-5 \
bash train.sh
```

adapter 默认写入 `models/adapters/qwen3.5-0.8b`。训练集和验证集的 loss 会由 MLX-LM 输出；建议根据验证 loss 调整迭代数，避免只依据训练 loss 延长训练。

## 合并并量化为 MLX

训练成功后运行：

```bash
bash export_mlx.sh
```

流程会先用 `mlx_lm.fuse` 将 LoRA adapter 合并到基础模型，再用 `mlx_lm.convert` 导出 4-bit MLX 模型。默认结果如下：

- Adapter：`models/adapters/qwen3.5-0.8b`
- 合并模型：`models/fused/qwen3.5-0.8b-lora`
- 量化模型：`models/mlx/qwen3.5-0.8b-lora-4bit`

可用 `MODEL`、`ADAPTER_PATH`、`FUSED_PATH`、`MLX_PATH` 和 `Q_BITS` 环境变量修改参数。导出脚本不会覆盖已有的合并目录或 MLX 目录；若要重新导出，请先自行移走或删除对应的目标目录。量化位数须为 MLX-LM 支持的值，默认 4-bit。

## 导入 oMLX

oMLX 通过扫描模型目录发现模型，而不是使用 `omlx import` 子命令。将导出的模型目录复制到 oMLX 当前配置的模型目录中；若你的模型目录是默认位置，可运行：

```bash
mkdir -p "$HOME/.omlx/models"
ditto models/mlx/qwen3.5-0.8b-lora-4bit \
  "$HOME/.omlx/models/qwen3.5-0.8b-lora-4bit"
```

如果你在 oMLX 中设置了其他模型目录，请将目标路径替换为该目录下一个新的模型子目录。然后刷新 oMLX 模型列表或重启服务。也可按 oMLX 当前版本的 CLI 启动服务并指定目录：

```bash
omlx serve --model-dir "$HOME/.omlx/models"
```

启动后用 `omlx --help` 和 oMLX 界面确认已发现模型，并按界面显示的模型 ID 调用。

如果 oMLX 报告模型或架构不可用，先确认 MLX 导出目录包含模型配置、tokenizer 文件和权重，并更新 oMLX 到支持 Qwen3.5 的版本。MLX 格式本身并不保证每个推理服务都支持该模型架构；Qwen3.5 在 oMLX 中的可用性也取决于 oMLX 版本和所需的模型内核。
