# OURS_Agent Cpp-NPU

## Video 数据流程
`MediaExtractor → MediaCodec → AImageReader → YUV→RGB`

## 编译ggml-hexagon
```bash
cd ~/path/to/llama.cpp
cmake -S . -B build-hexagon \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DGGML_HEXAGON=ON \
  -DGGML_OPENCL=OFF \
  -DGGML_OPENMP=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_CURL=OFF \
  -DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
  -DHEXAGON_TOOLS_ROOT=$HEXAGON_TOOLS_ROOT \
  -DPREBUILT_LIB_DIR=android_aarch64
cmake --build build-hexagon -j8
cmake --install build-hexagon \
  --prefix pkg-android/llama.cpp
```

## 编译 ouagent
```bash
cd ~/OURS_Agent
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DLLAMA_BIN_DIR="$PWD/third_party/llama.cpp/build-hexagon/bin"

cmake --build build-android --target ours_streamingbench -j8
```

## Android shell
```bash
./bin/ours_streamingbench --video /sdcard/Download/StreamingBench/video/sample_51/video.mp4 --model /data/local/tmp/ouagent/models/gemma-4-E4B-it-Q8_0.gguf --mmproj /data/local/tmp/ouagent/models/mmproj-gemma-4-E4B-it-Q8_0.gguf --csv /data/local/tmp/ouagent/testdata/StreamingBench/annotations/Real_Time_Visual_Understanding.csv --n-ctx 2048 --limit 1 --verbose
```
