# OURS_Agent: Streaming Video VLM with Persistent KV Cache

`OURS_Agent` 是一个基于 Gemma-4 (GGUF) 和 llama.cpp (libmtmd) 的**流式视频多模态推理框架**。
本项目实现了持久化 Session 与增量 KV Cache 追加，支持多视频帧的统一编码、边看边答，以及在生成中途实时注入新帧。

---

## 1. 环境要求与依赖准备

### 1.1 基础硬件与软件
- **GPU**: NVIDIA RTX  5090 
- **CUDA**: 推荐 12.8 及以上
- **Python**: 3.11（使用 Conda 环境）
- **C++ 编译器**: `gcc/g++ >= 11`，`CMake >= 3.20`

### 1.2 验证环境
```bash
nvcc --version
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```
> 确认输出中 CUDA 可用，且 PyTorch 的 CUDA 版本与系统一致（如 PyTorch 2.7.1+cu128）。

### 1.3 安装 Python 依赖
```bash
conda create -n ouagent python=3.10 -y
conda activate ouagent
pip install -r requirements.txt
```
## 2. 编译 llama.cpp
```bash
mkdir -p third_party && cd third_party
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# 开启 CUDA 支持，生成构建脚本
cmake -B build -DGGML_CUDA=ON

# 编译 
cmake --build build --config Release -j$(nproc)
```
> 验证编译是否成功
```bash
ls -lh build/bin/llama-mtmd-cli
```

## 3.下载模型文件
```bash
#  本项目需要 Gemma-4 的文本主模型和视觉投影器（mmproj），建议放在 models/gemma-4-E4B 目录下。

mkdir -p models/gemma-4-E4B
cd models/gemma-4-E4B

# 确保安装了 huggingface_hub
pip install -U huggingface_hub

# 下载文本主模型 (Q8_0 格式)
hf download ggml-org/gemma-4-E4B-GGUF gemma-4-E4B-Q8_0.gguf --local-dir .

# 下载视觉投影器 (mmproj)
curl -L -C - -o mmproj-F16.gguf "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf"
```
> 下载后，请确认 configs/gemma4.yaml 中的 model.path 与 model.mmproj 字段指向上述两个真实文件。

## 4. 快速测试与验证
### 4.1 官方CLI 验证单图
```bash
# 在 /path/to/OURS_Agent/third_party/llama.cpp 目录下运行：
./build/bin/llama-mtmd-cli \
  -m ~/OURS_Agent/models/gemma-4-E4B/gemma-4-E4B-Q8_0.gguf \
  --mmproj ~/OURS_Agent/models/gemma-4-E4B/mmproj-F16.gguf \
  --image ~/OURS_Agent/data/figure/COCO_val2014_000000210789.jpg \
  --chat-template gemma \
  -p "What color is the umbrella?" \
  -n 20 -ngl 99 --flash-attn on
```
### 4.2 运行本项目脚本
```bash
# 单图推理
python scripts/run_image.py \
  --config configs/gemma4.yaml \
  --image data/figure/COCO_val2014_000000210789.jpg \
  --prompt "What is the color of the umbrella?"
```
```bash
# 长视频（不是流式）
python scripts/run_video.py \
  --config configs/gemma4.yaml \
  --video /home/user/Dataset/video.mkv
```
### 4.3 streaming video inference
```bash
# 下载数据集
# 下载视频文件 (解压到 data/StreamingBench/video/)
hf download "hf://datasets/mjuicem/StreamingBench/Real-Time Visual Understanding_51-100.zip"

# 下载标注文件 (CSV)
hf download mjuicem/StreamingBench "Real-Time Visual Understanding.csv" \
  --repo-type dataset \
  --local-dir data/StreamingBench/annotations

# 批量
    python scripts/run_streamingbench_test.py --video-dir data/StreamingBench/video
    # 单视频
    python scripts/run_streamingbench_test.py --video data/StreamingBench/video/sample_55/video.mp4
```
> 注意：测试结果将生成在 results/streamingbench/<video_stem>/ 目录下，包含每个视频的 query 延迟、视觉 token 统计、KV Cache 增长等图表源数据 (CSV/JSON)。

