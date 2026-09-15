#!/bin/bash
# Harmony 环境安装脚本 (Apple Silicon)
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== Harmony 语音克隆环境安装 ==="

# 1. 初始化 git submodule
echo "[1/5] 初始化 GPT-SoVITS 子模块..."
git submodule update --init --depth=1 vendor/GPT-SoVITS

# 2. 创建 conda 环境
echo "[2/5] 创建 conda 环境 harmony (Python 3.11)..."
if conda env list | grep -q "^harmony "; then
    echo "conda 环境 harmony 已存在，跳过创建"
else
    conda create -n harmony python=3.11 -y
fi

# 3. 激活环境并安装依赖
echo "[3/5] 安装 PyTorch 和项目依赖..."
echo "请手动执行以下命令（conda activate 无法在脚本中运行）："
echo ""
echo "  conda activate harmony"
echo "  pip install torch torchaudio torchcodec"
echo "  pip install -r vendor/GPT-SoVITS/requirements.txt"
echo "  pip install pyyaml soundfile"
echo "  pip install addict datasets simplejson sortedcontainers  # modelscope 未声明的依赖"
echo "  pip install 'pypinyin>=0.53.0'                          # GPT-SoVITS 需要 to_finals_tone3"
echo "  conda install -c conda-forge ffmpeg -y"
echo ""

# 4. 下载预训练模型
echo "[4/5] 预训练模型下载说明..."
echo ""
echo "需要手动下载以下模型到 vendor/GPT-SoVITS/GPT_SoVITS/pretrained_models/："
echo ""
echo "  方式一 (推荐): 使用 huggingface-cli"
echo "    pip install huggingface_hub"
echo "    huggingface-cli download lj1995/GPT-SoVITS --local-dir vendor/GPT-SoVITS/GPT_SoVITS/pretrained_models"
echo ""
echo "  方式二: 手动下载"
echo "    https://huggingface.co/lj1995/GPT-SoVITS"
echo "    下载后解压到 vendor/GPT-SoVITS/GPT_SoVITS/pretrained_models/"
echo ""
echo "  需要的模型文件："
echo "    - chinese-roberta-wwm-ext-large/ (BERT)"
echo "    - chinese-hubert-base/ (CNHuBERT)"
echo "    - gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
echo "    - gsv-v2final-pretrained/s2G2333k.pth"
echo "    - gsv-v2final-pretrained/s2D2333k.pth"
echo ""

# 5. 下载语言检测模型
echo "[5/6] 下载 fast-langdetect 模型..."
LANGDETECT_DIR="vendor/GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect"
mkdir -p "$LANGDETECT_DIR"
if [ ! -f "$LANGDETECT_DIR/lid.176.bin" ]; then
    echo "下载 lid.176.bin (125MB)..."
    curl -L -o "$LANGDETECT_DIR/lid.176.bin" \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
else
    echo "lid.176.bin 已存在，跳过"
fi

# 6. 创建数据目录
echo "[6/6] 创建数据目录..."
mkdir -p data/raw data/sliced data/denoised data/asr_output
mkdir -p models/pretrained models/trained
mkdir -p output logs

echo ""
echo "=== 安装完成 ==="
echo "下一步："
echo "  1. conda activate harmony"
echo "  2. 安装上述 pip 依赖"
echo "  3. 下载预训练模型"
echo "  4. 将音频放入 data/raw/"
echo "  5. python cli.py pipeline"
