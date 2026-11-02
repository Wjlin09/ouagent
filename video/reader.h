// Video frame source.
//
// (libav*), which is already a system dependency of llama.cpp's MTMD_VIDEO
// build on the inference server. Frames are produced as RGB (ffmpeg RGB24).

#ifndef OURS_NATIVE_VIDEO_READER_H
#define OURS_NATIVE_VIDEO_READER_H

#include "common.h"

#include <cstdint>
#include <memory>
#include <string>

namespace ours {

struct VideoFrame {
    int64_t index = 0;
    double timestamp = 0.0;  // seconds on the video timeline
    Image image;             // RGB image
};

// Lazy frame source: yields frames one at a time without buffering the video.
class FrameSource {
public:
    virtual ~FrameSource() = default;
    // Fill ``out`` with the next frame; return false at end-of-stream.
    virtual bool next(VideoFrame & out) = 0;
};

// Reads decoded frames from a video file.
class VideoReader : public FrameSource {
public:
    explicit VideoReader(std::string path);
    ~VideoReader() override;

    VideoReader(const VideoReader &) = delete;
    VideoReader & operator=(const VideoReader &) = delete;

    // Returns false (and fills ``err``) if the file cannot be opened / decoded.
    bool open(std::string & err);
    bool is_open() const { return opened_; }
    bool next(VideoFrame & out) override;

    // Video duration in seconds (0.0 if the container reports none). Used by the
    // StreamingBench driver for the ``video_duration`` summary field; mirrors the
    // Python ``_video_duration_seconds`` helper without needing a second decoder.
    double duration() const { return duration_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string path_;
    bool opened_ = false;
    double duration_ = 0.0;
};

} // namespace ours

#endif // OURS_NATIVE_VIDEO_READER_H
