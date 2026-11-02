# OURS_Agent native (C++) build.
#
# Builds the full C++ Agent, not just the runtime:
#   * libours_gemma_runtime.so  -- GemmaRuntime + C ABI (the JNI seam)
#   * ours_agent                 -- the C++ Agent executable (full path)
#   * ours_runtime_cli           -- low-level runtime smoke driver (NOT the Agent)
#
# Links against *pre-built* libllama/libmtmd (do NOT rebuild llama.cpp here).

cmake_minimum_required(VERSION 3.16)
project(ours_agent_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vendored llama.cpp source + built binaries. Defaults assume the server layout;
# override with -DLLAMA_CPP_DIR=... / -DLLAMA_BUILD_DIR=... if needed.
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp" CACHE PATH
    "Path to the llama.cpp source tree (for headers).")
set(LLAMA_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/build" CACHE PATH
    "Path to the built llama.cpp (for libllama.so / libmtmd.so).")

set(OURS_INCLUDE_DIRS
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/tools/mtmd
)

find_library(OURS_LLAMA_LIBRARY NAMES llama
    PATHS ${LLAMA_BUILD_DIR}/bin ${LLAMA_BUILD_DIR} NO_DEFAULT_PATH)
find_library(OURS_MTMD_LIBRARY NAMES mtmd
    PATHS ${LLAMA_BUILD_DIR}/bin ${LLAMA_BUILD_DIR} NO_DEFAULT_PATH)
find_library(
    OURS_GGML_BASE_LIBRARY
    NAMES ggml-base
    PATHS ${LLAMA_BUILD_DIR}/bin ${LLAMA_BUILD_DIR}
    NO_DEFAULT_PATH
)

if(NOT OURS_LLAMA_LIBRARY OR NOT OURS_MTMD_LIBRARY)
    message(FATAL_ERROR
        "libllama/libmtmd not found under LLAMA_BUILD_DIR=${LLAMA_BUILD_DIR}. "
        "Build llama.cpp on the Linux server first (out of scope here), then "
        "configure with -DLLAMA_BUILD_DIR=<path-to-llama.cpp-build>.")
endif()

# ---------------------------------------------------------------------------
# Runtime library: GemmaRuntime + C ABI (JNI seam)
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
target_link_libraries(ours_gemma_runtime PRIVATE ${OURS_MTMD_LIBRARY} ${OURS_LLAMA_LIBRARY})
target_link_directories(ours_gemma_runtime PRIVATE ${LLAMA_BUILD_DIR}/bin)
set_target_properties(ours_gemma_runtime PROPERTIES
    BUILD_RPATH "${LLAMA_BUILD_DIR}/bin"
    INSTALL_RPATH "${LLAMA_BUILD_DIR}/bin"
)

# ---------------------------------------------------------------------------
# ffmpeg (libav) for video/reader.cpp -- optional; llama.cpp MTMD_VIDEO already
# requires it on the server, so it is normally present.
# ---------------------------------------------------------------------------
set(OURS_FFMPEG_DEFS "")
set(OURS_FFMPEG_LIBS "")

#set(OURS_RABBITMQ_LIBRARY
#    /usr/lib/x86_64-linux-gnu/librabbitmq.so.4
#)
#
#set(OURS_SSH_LIBRARY
#    /usr/lib/x86_64-linux-gnu/libssh-gcrypt.so.4
#)
#
#set(OURS_VA_LIBRARY
#    /usr/lib/x86_64-linux-gnu/libva.so.2
#)
#
#set(OURS_DAV1D_LIBRARY
#    /usr/lib/x86_64-linux-gnu/libdav1d.so.5
#)
#
#set(OURS_OPUS_LIBRARY
#    /usr/lib/x86_64-linux-gnu/libopus.so.0
#)
#
#set(OURS_FFMPEG_EXTRA_LIBS
#    ${OURS_RABBITMQ_LIBRARY}
#    ${OURS_SSH_LIBRARY}
#    ${OURS_VA_LIBRARY}
#    ${OURS_DAV1D_LIBRARY}
#    ${OURS_OPUS_LIBRARY}
#)

#find_package(PkgConfig QUIET)
#if(PkgConfig_FOUND)
#    pkg_check_modules(OURS_FFMPEG QUIET IMPORTED_TARGET
#        libavformat libavcodec libswscale libavutil)
#    if(OURS_FFMPEG_FOUND)
#        set(OURS_FFMPEG_DEFS OURS_HAVE_FFMPEG)
#        set(OURS_FFMPEG_LIBS PkgConfig::OURS_FFMPEG)
#
#        # libavformat/libavcodec carry private transitive deps that modern ld
#        # (--no-copy-dt-needed-entries) does not resolve from a shared library's
#        # DT_NEEDED when linking the final executable:
#        #   * libavformat -> libssh  (SFTP:  ssh_new)
#        #   * libavcodec  -> libva   (VAAPI: vaBeginPicture / vaDestroyConfig)
#        # Discover them and add them to the executable link line. libssh is
#        # sometimes installed as its gcrypt variant (libssh-gcrypt.so), so probe
#        # both names.
#        #find_library(OURS_SSH_LIBRARY NAMES ssh ssh-gcrypt)
#        #find_library(OURS_VA_LIBRARY NAMES va)
#        #if(OURS_SSH_LIBRARY)
#        #    list(APPEND OURS_FFMPEG_EXTRA_LIBS ${OURS_SSH_LIBRARY})
#        #endif()
#        #if(OURS_VA_LIBRARY)
#        #    list(APPEND OURS_FFMPEG_EXTRA_LIBS ${OURS_VA_LIBRARY})
#        #endif()
#    endif()
#endif()
find_package(PkgConfig REQUIRED)

pkg_check_modules(OURS_FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libswscale libavutil)

set(OURS_FFMPEG_DEFS OURS_HAVE_FFMPEG)
set(OURS_FFMPEG_LIBS PkgConfig::OURS_FFMPEG)
if(NOT OURS_FFMPEG_FOUND)
    message(WARNING
        "ffmpeg dev libs not found via pkg-config; video/reader.cpp will build "
        "as a no-decode stub. ffmpeg/libav dev headers+libs are a missing "
        "dependency for real video decoding -- report it, do not install.")
endif()

# ---------------------------------------------------------------------------
# C++ Agent executable (full path: Agent -> Controller/State -> Video/Algo ->
# GemmaRuntime -> libmtmd/libllama)
# ---------------------------------------------------------------------------
add_executable(ours_agent
    main.cpp
    agent/agent.cpp
    video/reader.cpp
)
target_include_directories(ours_agent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OURS_INCLUDE_DIRS}
)
target_compile_features(ours_agent PRIVATE cxx_std_17)
target_compile_definitions(ours_agent PRIVATE ${OURS_FFMPEG_DEFS})
target_link_libraries(ours_agent PRIVATE
    ours_gemma_runtime
    ${OURS_LLAMA_LIBRARY} ${OURS_MTMD_LIBRARY}
    ${OURS_GGML_BASE_LIBRARY}
    ${OURS_FFMPEG_LIBS} 
    #${OURS_FFMPEG_EXTRA_LIBS}
)
target_link_directories(ours_agent PRIVATE ${LLAMA_BUILD_DIR}/bin)
set_target_properties(ours_agent PROPERTIES BUILD_RPATH "${LLAMA_BUILD_DIR}/bin")

# ---------------------------------------------------------------------------
# StreamingBench real-time driver (single video). Reuses the full Agent path;
# one MP4 = one persistent Agent session, generation interleaved with ingestion.
# ---------------------------------------------------------------------------
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
target_compile_definitions(ours_streamingbench PRIVATE ${OURS_FFMPEG_DEFS})
target_link_libraries(ours_streamingbench PRIVATE
    ours_gemma_runtime
    ${OURS_LLAMA_LIBRARY} ${OURS_MTMD_LIBRARY}
    ${OURS_GGML_BASE_LIBRARY}
    ${OURS_FFMPEG_LIBS}
)
target_link_directories(ours_streamingbench PRIVATE ${LLAMA_BUILD_DIR}/bin)
set_target_properties(ours_streamingbench PROPERTIES BUILD_RPATH "${LLAMA_BUILD_DIR}/bin")

# ---------------------------------------------------------------------------
# Low-level runtime smoke driver (kept; NOT a substitute for the Agent)
# ---------------------------------------------------------------------------
add_executable(ours_runtime_cli ours_runtime_cli.cpp)
target_include_directories(ours_runtime_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OURS_INCLUDE_DIRS}
)
target_compile_features(ours_runtime_cli PRIVATE cxx_std_17)
target_link_libraries(ours_runtime_cli PRIVATE
    ours_gemma_runtime
    ${OURS_LLAMA_LIBRARY}
    ${OURS_MTMD_LIBRARY}
    ${OURS_GGML_BASE_LIBRARY}
)
target_link_directories(ours_runtime_cli PRIVATE ${LLAMA_BUILD_DIR}/bin)
set_target_properties(ours_runtime_cli PROPERTIES BUILD_RPATH "${LLAMA_BUILD_DIR}/bin")
