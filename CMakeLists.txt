# OURS_Agent native (C++) build.
#
# Host (Linux) builds the full C++ Agent:
#   * libours_gemma_runtime.so  -- GemmaRuntime + C ABI (the JNI seam)
#   * ours_agent                 -- the C++ Agent executable (full path)
#   * ours_streamingbench        -- real-time single-video driver
#   * ours_runtime_cli           -- low-level runtime smoke driver (NOT the Agent)
#
# Android ARM64 cross-compile builds ONLY the runtime:
#   * libours_gemma_runtime.so  -- GemmaRuntime + C ABI (the JNI seam)
#   (no host CLI executables, no Linux FFmpeg, no host video reader)
#
# Links against *pre-built* libllama/libmtmd/libggml-* (do NOT rebuild llama.cpp
# here). Prebuilt libs are imported as IMPORTED targets -- never
# link_directories()/link_libraries().

cmake_minimum_required(VERSION 3.16)
project(ours_agent_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Vendored llama.cpp source (headers) + prebuilt binaries.
# ---------------------------------------------------------------------------
# No absolute paths are hard-coded here. Everything is overridable from the
# configure line with -D (see the build commands below):
#   LLAMA_CPP_DIR : llama.cpp source tree (for headers)
#   LLAMA_BIN_DIR : directory holding the prebuilt .so files
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp" CACHE PATH
    "Path to the llama.cpp source tree (for headers).")

if(ANDROID)
    # Hexagon Android build on the server: prebuilt libs live under build-hexagon/bin.
    set(LLAMA_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/build-hexagon/bin" CACHE PATH
        "Directory with prebuilt libllama.so / libmtmd.so / libggml-*.so.")
else()
    set(LLAMA_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/build/bin" CACHE PATH
        "Directory with prebuilt libllama.so / libmtmd.so / libggml-*.so.")
endif()

set(OURS_INCLUDE_DIRS
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/tools/mtmd
)

# ---------------------------------------------------------------------------
# Import the prebuilt shared libraries as IMPORTED targets.
# ---------------------------------------------------------------------------
add_library(ours_llama SHARED IMPORTED)
set_target_properties(ours_llama PROPERTIES
    IMPORTED_LOCATION "${LLAMA_BIN_DIR}/libllama.so")

add_library(ours_mtmd SHARED IMPORTED)
set_target_properties(ours_mtmd PROPERTIES
    IMPORTED_LOCATION "${LLAMA_BIN_DIR}/libmtmd.so")

add_library(ours_ggml_base SHARED IMPORTED)
set_target_properties(ours_ggml_base PROPERTIES
    IMPORTED_LOCATION "${LLAMA_BIN_DIR}/libggml-base.so")

# CPU-side Hexagon backend driver + the rest of the ggml stack. These exist
# only in the Hexagon Android build.
# NOTE: libggml-htp-v75.so (the DSP-side compute library) is intentionally NOT
# imported or linked here; it is loaded at runtime via ADSP_LIBRARY_PATH.
if(ANDROID)
    add_library(ours_ggml_hexagon SHARED IMPORTED)
    set_target_properties(ours_ggml_hexagon PROPERTIES
        IMPORTED_LOCATION "${LLAMA_BIN_DIR}/libggml-hexagon.so")

    add_library(ours_ggml SHARED IMPORTED)
    set_target_properties(ours_ggml PROPERTIES
        IMPORTED_LOCATION "${LLAMA_BIN_DIR}/libggml.so")

    add_library(ours_ggml_cpu SHARED IMPORTED)
    set_target_properties(ours_ggml_cpu PROPERTIES
        IMPORTED_LOCATION "${LLAMA_BIN_DIR}/libggml-cpu.so")
endif()

# Fail early with a clear message if a required prebuilt lib is missing.
foreach(lib IN ITEMS ours_llama ours_mtmd ours_ggml_base)
    get_target_property(_ours_loc ${lib} IMPORTED_LOCATION)
    if(NOT _ours_loc OR NOT EXISTS "${_ours_loc}")
        message(FATAL_ERROR
            "${lib} not found at '${_ours_loc}'. Build llama.cpp first, or "
            "override with -DLLAMA_BIN_DIR=<dir-with-prebuilt-libs>.")
    endif()
endforeach()
if(ANDROID)
    foreach(lib IN ITEMS ours_ggml_hexagon ours_ggml ours_ggml_cpu)
        get_target_property(_ours_loc ${lib} IMPORTED_LOCATION)
        if(NOT _ours_loc OR NOT EXISTS "${_ours_loc}")
            message(FATAL_ERROR
                "${lib} not found at '${_ours_loc}'. Build the Hexagon "
                "llama.cpp target first, or override with -DLLAMA_BIN_DIR=<dir>.")
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Runtime library: GemmaRuntime + C ABI (JNI seam).
# The ONLY target built for Android.
# ---------------------------------------------------------------------------
add_library(ours_gemma_runtime SHARED
    runtime/gemma_runtime.cpp
    runtime/gemma_runtime_c_api.cpp
)
target_include_directories(ours_gemma_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OURS_INCLUDE_DIRS}
)
target_compile_features(ours_gemma_runtime PRIVATE cxx_std_17)
target_compile_definitions(ours_gemma_runtime PRIVATE OURS_RUNTIME_BUILD)
target_link_libraries(ours_gemma_runtime PRIVATE
    ours_mtmd
    ours_llama
    ours_ggml_base
)
if(ANDROID)
    target_link_libraries(ours_gemma_runtime PRIVATE
        ours_ggml
        ours_ggml_cpu
        ours_ggml_hexagon
    )
    # Keep the CPU-side Hexagon backend driver on the link line even though our
    # runtime references none of its symbols directly (libllama resolves it via
    # the ggml backend registry); otherwise --as-needed would drop it.
    target_link_options(ours_gemma_runtime PRIVATE "-Wl,--no-as-needed")
endif()

# ---------------------------------------------------------------------------
# Host-only CLI drivers (Linux native). NOT built for Android.
# ---------------------------------------------------------------------------
if(NOT ANDROID)
    # Host video reader: a stub that keeps the VideoReader interface but cannot
    # decode. The real Android mediandk reader (video/reader.cpp) is consumed by
    # the future JNI/APK layer, not by these host drivers.
    set(OURS_VIDEO_SOURCE video/reader_stub.cpp)

    # C++ Agent executable (full path: Agent -> Controller/State -> Video/Algo ->
    # GemmaRuntime -> libmtmd/libllama)
    add_executable(ours_agent
        main.cpp
        agent/agent.cpp
        ${OURS_VIDEO_SOURCE}
    )
    target_include_directories(ours_agent PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OURS_INCLUDE_DIRS}
    )
    target_compile_features(ours_agent PRIVATE cxx_std_17)
    target_link_libraries(ours_agent PRIVATE
        ours_gemma_runtime
        ours_llama
        ours_mtmd
        ours_ggml_base
    )
    set_target_properties(ours_agent PROPERTIES BUILD_RPATH "${LLAMA_BIN_DIR}")

    # StreamingBench real-time driver (single video). Reuses the full Agent path;
    # one MP4 = one persistent Agent session, generation interleaved with ingestion.
    add_executable(ours_streamingbench
        ours_streamingbench.cpp
        agent/agent.cpp
        ${OURS_VIDEO_SOURCE}
    )
    target_include_directories(ours_streamingbench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OURS_INCLUDE_DIRS}
    )
    target_compile_features(ours_streamingbench PRIVATE cxx_std_17)
    target_link_libraries(ours_streamingbench PRIVATE
        ours_gemma_runtime
        ours_llama
        ours_mtmd
        ours_ggml_base
    )
    set_target_properties(ours_streamingbench PROPERTIES BUILD_RPATH "${LLAMA_BIN_DIR}")

    # Low-level runtime smoke driver (kept; NOT a substitute for the Agent)
    add_executable(ours_runtime_cli ours_runtime_cli.cpp)
    target_include_directories(ours_runtime_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OURS_INCLUDE_DIRS}
    )
    target_compile_features(ours_runtime_cli PRIVATE cxx_std_17)
    target_link_libraries(ours_runtime_cli PRIVATE
        ours_gemma_runtime
        ours_llama
        ours_mtmd
        ours_ggml_base
    )
    set_target_properties(ours_runtime_cli PROPERTIES BUILD_RPATH "${LLAMA_BIN_DIR}")
endif()

# ---------------------------------------------------------------------------
# Android ARM64: StreamingBench native executable.
# Uses the real Android NDK video reader (video/reader.cpp) instead of the host
# stub, and links libmediandk plus the Hexagon prebuilt ggml stack.
# ---------------------------------------------------------------------------
if(ANDROID)
    find_library(OURS_ANDROID_MEDIA_LIBRARY NAMES mediandk)
    if(NOT OURS_ANDROID_MEDIA_LIBRARY)
        message(FATAL_ERROR
            "libmediandk (Android NDK media) not found; required by video/reader.cpp.")
    endif()

    add_executable(ours_streamingbench
        ours_streamingbench.cpp
        agent/agent.cpp
        video/reader.cpp
    )
    target_include_directories(ours_streamingbench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OURS_INCLUDE_DIRS}
    )
    target_compile_features(ours_streamingbench PRIVATE cxx_std_17)
    target_link_libraries(ours_streamingbench PRIVATE
        ours_gemma_runtime
        ours_llama
        ours_mtmd
        ours_ggml_base
        ours_ggml
        ours_ggml_cpu
        ours_ggml_hexagon
        ${OURS_ANDROID_MEDIA_LIBRARY}
    )
    # Keep the CPU-side Hexagon backend driver on the link line (same rationale
    # as ours_gemma_runtime): libllama resolves it via the ggml backend registry.
    target_link_options(ours_streamingbench PRIVATE "-Wl,--no-as-needed")
endif()
