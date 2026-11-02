// Streaming frame sampler.
//
// Forwards at most ``fps`` frames per second, capped at ``max_frames`` in total.
// Pull-based and stateful, so the Agent can interleave sampling with generation
// without buffering the whole video.

#ifndef OURS_NATIVE_VIDEO_SAMPLER_H
#define OURS_NATIVE_VIDEO_SAMPLER_H

#include "video/reader.h"

#include <utility>

namespace ours {

class FrameSampler {
public:
    // ``max_frames < 0`` means uncapped (stream the whole input).
    explicit FrameSampler(double fps = 2.0, int max_frames = 16)
        : interval_(fps > 0.0 ? 1.0 / fps : 0.0), max_frames_(max_frames) {}

    // Pull the next *selected* frame from ``src``. Returns false when done
    // (max_frames reached or source exhausted). Does not drain ``src`` past the
    // cap -- mirrors the Python ``stream()`` generator.
    bool next(FrameSource & src, VideoFrame & out) {
        while (true) {
            if (max_frames_ >= 0 && sampled_ >= max_frames_) {
                return false;
            }
            VideoFrame f;
            if (!src.next(f)) {
                return false;
            }
            if (interval_ > 0.0 && f.timestamp < next_timestamp_) {
                continue;
            }
            ++sampled_;
            next_timestamp_ = f.timestamp + interval_;
            out = std::move(f);
            return true;
        }
    }

    int sampled() const { return sampled_; }

private:
    double interval_;
    double next_timestamp_ = 0.0;
    int max_frames_;
    int sampled_ = 0;
};

} // namespace ours

#endif // OURS_NATIVE_VIDEO_SAMPLER_H
