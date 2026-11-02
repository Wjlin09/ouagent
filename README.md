# OURS_Agent native (C++) — Full Agent Migration
C++ 版本的 OURS_Agent 运行时。项目直接调用原生的 libmtmd + libllama C API。
GPU: NVIDIA RTX 5090
CUDA: 推荐 12.8 及以上
C++ 编译器: gcc/g++ >= 11，CMake >= 3.20

# 1. 目录结构 (Module Layout)
```text
OURS_Agent/
├── CMakeLists.txt              # 构建脚本 (链接 libllama, libmtmd, ffmpeg)
├── main.cpp                    # [入口 1] ours_agent：单视频、单 Query 的简单驱动
├── ours_streamingbench.cpp     # [入口 2] ours_streamingbench：流式测试驱动（核心）
├── ours_runtime_cli.cpp        # [入口 3] ours_runtime_cli：底层 Runtime 冒烟测试
│
├── agent/                      # 业务编排层
│   ├── agent.h / .cpp          # Agent：封装原子操作 (Ingest, StartQuery, StepGen)
│   ├── controller.h            # 策略决策：根据 State 决定下一步 Action
│   └── state.h                 # 状态机：Session 生命周期与 Query 生命周期的独立管理
│
├── runtime/                    # 底层推理层
│   ├── gemma_runtime.h / .cpp  # C++ 原生推理封装（驱动 llama.cpp）
│   └── gemma_runtime_c_api.h/.cpp # C ABI 接口（为未来 Android/JNI/NPU 预留）
│
├── video/                      # 视频流处理层
│   ├── reader.h / .cpp         # 视频解码 (基于 ffmpeg)，输出 RGB 帧与时间戳
│   └── sampler.h               # 帧采样器 (控制 fps 与 max_frames 抽帧)
│
├── algorithms/                 # 算法扩展层 (预留)
│   ├── frame_selector.h        # 视觉帧过滤（如 AutoGaze）
│   ├── semantic_pruning.h      # 语义剪枝
│   └── ...                     # 其它 Token/Region 选择器接口
│
└── evaluation/                 
    └── latency.h               # 延迟统计
```
本项目的核心原则是 Session 生命周期包含多个Query 生命周期，以 chunk 为单位进行处理，部分测试结果存储在results。
Session：对应一整段视频，负责维护持久化的 KV Cache。
Query：对应视频时间线上的单个问题，stop_query 或回答完毕不清空 Session 的 KV Cache。

# 2. 环境依赖与编译顺序 (Build Order)
本项目依赖预先编译好的 libllama.so、libmtmd.so 以及 ffmpeg 开发库。

## 第一步：准备并编译 llama.cpp
首先，在 OURS_Agent 目录下拉取 llama.cpp 并编译它（生成动态链接库）：
llama-cpp 部署 & 编译

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/ggml-org/llama.cpp.git
```

```bash
cd llama.cpp
cmake -B build -DGGML_CUDA=ON #检查环境，生成构建脚本，假设已经配置好了 CUDA 环境
```

```bash
cmake --build build --config Release -j$(nproc) 
# **编译阶段**，调用底层构建工具（make/ninja），实际调用编译器去翻译代码，最后生成 `llama-mtmd-cli`
# cmake --build build --config Release -j4
```
> 检查结果

```bash
ls -lh build/bin/llama-mtmd-cli # 可以看到这个文件说明编译成功
# ls -lh ~/OURS_Agent/third_party/llama.cpp/build/bin/llama-mtmd-cli
```
## 第二步：准备Gemma4 文件
https://gemma.ai/get-started/
>`mtmd` 的调用需要模型文件和 `mmproj`；
>官方 `llama-mtmd-cli` 的接口就是通过 `-m/--model` 和 `--mmproj` 加载它们，再通过 `--image` 输入图片

```bash
mkdir -p models/gemma-4-E4B
cd models/gemma-4-E4B
```

```bash
# hf version 
#有就行了，没有再执行以下命令
pip install -U huggingface_hub # 安装 Hugging Face CLI
```

```bash
hf download ggml-org/gemma-4-E4B-GGUF \
    gemma-4-E4B-Q8_0.gguf \
    --local-dir .
    
cd ~/OURS_Agent/models/gemma-4-E4B

curl -L -C - \
  -o mmproj-F16.gguf \
  "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf"
```

# 3. 在 OURS_Agent 目录下编译项目
```bash
cmake -B build .
cmake --build build -j8
#(注：如果 llama.cpp 的编译产物不在默认的 third_party/llama.cpp/build 路径下，可以通过 -DLLAMA_CPP_DIR=... 和 -DLLAMA_BUILD_DIR=... 参数覆盖)
```

# 4. 运行测试 (Usage)
以下所有命令假设处于 OURS_Agent 目录。
## 第一步：拉取 Streamingbench 数据集
```bash
cd data/StreamingBench/video/
hf download "hf://datasets/mjuicem/StreamingBench/Real-Time Visual Understanding_51-100.zip"

#下载标注文件 (CSV)
hf download mjuicem/StreamingBench "Real-Time Visual Understanding.csv" \
  --repo-type dataset \
  --local-dir data/StreamingBench/annotations
```
## 第二步：单视频（非流式测试）

```bash
./build/ours_agent \
  --video ./data/StreamingBench/video/sample_51/video.mp4 \
  --model ./models/gemma-4-E4B/gemma-4-E4B-it-Q8_0.gguf \
  --mmproj ./models/gemma-4-E4B/mmproj-gemma-4-E4B-it-Q8_0.gguf \
  --prompt "What is happening in the video?" \
  --max-frames 16 \
  --fps 1.0 \
  --resize 336 \
  --verbose
  ```
## 第三步： StreamingBench 测试 (ours_streamingbench)
针对StreamingBench 的连续视频流测试（持久化 Session，按视频时间戳触发 Query）。
### 参数设置
当前支持的参数包括：

| 参数 | StreamingBench 默认值 |
|---|---:|
| `temperature` | `0.0` |
| `top_p` | `0.95` |
| `repeat_penalty` | `1.1` |
| `seed` | `1234` |
| `max_tokens` | `1024` |
| `n_ctx` | `65536` |
| `fps` | `1.0` |
| `resize` | `336` |
| `num_frameproc` | `4` |

> num_frameproc: 每次批处理（Chunk）的帧数。决定 ingest_frame 攒够多少帧才一次性注入 KV Cache 中。影响处理效率和显存的瞬时峰值。

### 单视频测试（限制 1 个问题排错）：
```bash
./build/ours_streamingbench \
  --video ./data/StreamingBench/video/sample_52/video.mp4 \
  --model ./models/gemma-4-E4B/gemma-4-E4B-it-Q8_0.gguf \
  --mmproj ./models/gemma-4-E4B/mmproj-gemma-4-E4B-it-Q8_0.gguf \
  --limit 1
  ```
### 单视频完整测试：
```bash
./build/ours_streamingbench \
  --video ./data/StreamingBench/video/sample_52/video.mp4 \
  --model ./models/gemma-4-E4B/gemma-4-E4B-it-Q8_0.gguf \
  --mmproj ./models/gemma-4-E4B/mmproj-gemma-4-E4B-it-Q8_0.gguf 
  ```
> 测试结果会存在 results 里面

### 多视频测试（空格分隔）：
```bash
./build/ours_streamingbench \
  --video ./data/StreamingBench/video/sample_55/video.mp4 ./data/StreamingBench/video/sample_56/video.mp4 \
  --model ./models/gemma-4-E4B/gemma-4-E4B-it-Q8_0.gguf \
  --mmproj ./models/gemma-4-E4B/mmproj-gemma-4-E4B-it-Q8_0.gguf 
  ```

### 批量目录测试（自动遍历目录下的 sample_NN/video.mp4）：
```bash
./build/ours_streamingbench \
  --video ./data/StreamingBench/video/ \
  --model ./models/gemma-4-E4B/gemma-4-E4B-it-Q8_0.gguf \
  --mmproj ./models/gemma-4-E4B/mmproj-gemma-4-E4B-it-Q8_0.gguf 
  ```


