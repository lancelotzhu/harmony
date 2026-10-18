# Harmony

Harmony 是一个基于 GPT-SoVITS 的语音克隆与文本转语音项目：用一段原始录音训练说话人模型，再输入新文本，生成接近原说话人音色的 WAV 音频。

请只使用本人或已获得明确授权的录音，不要用生成语音冒充他人。

## 工作流程

1. 准备说话人清晰、背景噪声少的录音，放入 `data/raw/`。
2. 运行预处理：切分音频、降噪并识别文本。
3. 检查 ASR 识别结果，修正错字后训练 SoVITS 和 GPT 模型。
4. 输入要说的新文本，并提供一段 3–10 秒的参考音频及其准确文本。
5. 在 `output/` 中取得生成的 WAV 文件。

参考音频用于合成时提供音色和提示文本；它不能代替本项目所需的说话人模型训练。项目可以直接从原录音中选取参考片段，也可以在合成命令中明确指定片段。

## 环境准备

本项目的 `setup_env.sh` 面向 Apple Silicon macOS 和 Python 3.11。先运行脚本：

```bash
bash setup_env.sh
```

脚本会初始化 GPT-SoVITS 子模块、创建 Conda 环境并输出依赖和模型下载说明。按脚本提示激活环境并安装依赖：

```bash
conda activate harmony
pip install torch torchaudio torchcodec
pip install -r vendor/GPT-SoVITS/requirements.txt
pip install pyyaml soundfile
pip install addict datasets simplejson sortedcontainers
pip install 'pypinyin>=0.53.0'
conda install -c conda-forge ffmpeg -y
```

还需要从 Hugging Face 下载 GPT-SoVITS 预训练模型。推荐使用：

```bash
pip install huggingface_hub
huggingface-cli download lj1995/GPT-SoVITS --local-dir vendor/GPT-SoVITS/GPT_SoVITS/pretrained_models
```

请确认以下文件或目录存在（目录根路径为 `vendor/GPT-SoVITS/GPT_SoVITS/pretrained_models/`）：

- `chinese-roberta-wwm-ext-large/`
- `chinese-hubert-base/`
- `gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt`
- `gsv-v2final-pretrained/s2G2333k.pth`
- `gsv-v2final-pretrained/s2D2333k.pth`

## 准备录音并训练

把目标说话人的录音放到 `data/raw/`，然后运行完整流程：

```bash
python cli.py pipeline
```

`pipeline` 会执行音频切分、降噪、ASR 标注、训练以及一次测试合成。训练数据会写入 `logs/<experiment_name>/`，训练好的权重会写入：

- `models/trained/SoVITS_weights_v2/`
- `models/trained/GPT_weights_v2/`

也可以分步执行，以便检查和修正识别文本：

```bash
python cli.py preprocess
```

查看 `data/asr_output/` 下生成的 `.list` 文件，修正其中不准确的识别文本，保留每行原有的字段和分隔符。然后将实际生成的标注文件路径传给训练命令，例如：

```bash
python cli.py train --asr-list data/asr_output/denoised.list
```

如果预处理跳过了降噪，标注文件可能是 `data/asr_output/sliced.list`，请按实际文件名替换。脚本未指定 `--asr-list` 时，会自动选择 `data/asr_output/` 中最近修改的 `.list` 文件。

## 输入文本生成语音

训练完成后，直接输入文本并指定参考音频及其准确文本：

```bash
python cli.py synthesize \
  "你好，这是用我的原音生成的一段新语音。" \
  --ref-audio data/denoised/你的参考片段.wav \
  --ref-text "参考片段中实际说出的内容。" \
  --output output/result.wav
```

参考音频建议清晰、单人、没有背景音乐，时长为 3–10 秒。`--ref-text` 必须与参考音频中的话一致；不确定时不要留空，准确提示文本通常有助于合成稳定。也可以使用 `data/sliced/` 中的片段；若文件不在 `data/denoised/`，将命令中的路径改成实际位置。

省略 `--output` 时，文件会以时间戳命名并保存到 `output/`。默认情况下，CLI 会自动选择 `models/trained/` 中最新的 GPT 和 SoVITS 权重。需要使用指定权重时，可增加：

```bash
--gpt-model models/trained/GPT_weights_v2/你的模型.ckpt \
--sovits-model models/trained/SoVITS_weights_v2/你的模型.pth
```

也可以从 UTF-8 文本文件读取内容：

```bash
python cli.py synthesize --text-file input.txt \
  --ref-audio data/denoised/你的参考片段.wav \
  --ref-text "参考片段中实际说出的内容。" \
  --output output/result.wav
```

文本超过 200 个字符时，CLI 会自动使用长文本合成流程。语言、设备、预处理、训练和合成参数可在 `config.yaml` 中调整；当前默认文本和提示音频语言为中文，设备为 Apple Silicon 的 `mps`。

## 常用命令

```bash
python cli.py preprocess --skip-denoise  # 跳过降噪
python cli.py preprocess --force         # 清除预处理缓存并重新处理
python cli.py train --prepare-only       # 只准备训练数据
python cli.py train --force              # 重新训练已有模型
python cli.py --help                     # 查看命令帮助
```

新增或替换 `data/raw/` 中的音频后，使用 `python cli.py pipeline --force` 可重新预处理并训练。默认缓存标记会跳过已完成的预处理步骤。

## 常见问题

- **提示缺少预训练模型**：检查“环境准备”列出的模型文件是否位于配置指定目录。
- **提示未找到训练好的模型**：先成功运行 `python cli.py train` 或 `python cli.py pipeline`，并确认两类权重目录中分别有 `.pth` 和 `.ckpt` 文件。
- **提示未找到参考音频**：显式传入 `--ref-audio` 和 `--ref-text`；参考音频时长需在 3–10 秒之间。
- **新录音未参与处理**：运行 `python cli.py pipeline --force` 清除缓存并重做预处理与训练。
- **合成音色或发音不理想**：检查录音是否清晰、ASR 文本是否已校正，以及参考音频和 `--ref-text` 是否匹配。