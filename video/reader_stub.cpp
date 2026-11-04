// Host-build stub for the Android NDK video reader. Keeps the same VideoReader
// interface so the host build still links, but cannot decode; the real
// implementation lives in video/reader.cpp (Android NDK mediandk), selected by
// CMake via OURS_VIDEO_SOURCE.

#include "video/reader.h"

#include <cstdio>
#include <string>
#include <utility>

namespace ours {

struct VideoReader::Impl {};

VideoReader::VideoReader(std::string path) : impl_(new Impl()), path_(std::move(path)) {}

VideoReader::~VideoReader() = default;

bool VideoReader::open(std::string & err) {
    const char * msg = "Video decoding is unavailable in the host build; "
                       "this build requires Android NDK mediandk.";
    std::fprintf(stderr, "%s\n", msg);
    err = msg;
    return false;
}

bool VideoReader::next(VideoFrame & out) {
    (void) out;
    return false;
}

} // namespace ours
