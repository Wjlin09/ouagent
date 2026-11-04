// VideoReader implementation 
#include "video/reader.h"

#include <android/native_window.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ours {

namespace {

// Number of decoded frames the AImageReader buffers before back-pressuring the
// codec. Small enough to keep latency low, large enough to avoid stalls.
constexpr int32_t kMaxReaderImages = 4;

// Bounded EOS-draining budget and the poll interval between feed/drain passes.
constexpr int64_t kEosDrainTimeoutMs = 400;
constexpr int64_t kPollIntervalMs = 5;

inline uint8_t clamp255(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : static_cast<uint8_t>(v));
}

// Monotonic wall clock in milliseconds; used to bound the EOS-drain phase.
inline int64_t steady_now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// YUV_420_888 -> packed RGB24 (BT.601, limited range). Returns true on success.
//
// Reads the plane layout through the NDK out-parameter API and bounds-checks
// every Y/U/V sample access against its plane's dataLength, so a truncated or
// invalid buffer never yields a partial RGB frame. Supports planar (chroma
// pixelStride == 1) and semi-planar (interleaved chroma) layouts without
// assuming a fixed U/V pixel stride.
bool yuv420_to_rgb(const AImage * image, uint8_t * rgb) {
    if (rgb == nullptr) return false;

    int32_t w = 0;
    int32_t h = 0;
    if (AImage_getWidth(image, &w) != AMEDIA_OK || w <= 0) return false;
    if (AImage_getHeight(image, &h) != AMEDIA_OK || h <= 0) return false;

    int32_t num_planes = 0;
    if (AImage_getNumberOfPlanes(image, &num_planes) != AMEDIA_OK || num_planes < 3) {
        return false;
    }

    uint8_t * planes[3] = {nullptr, nullptr, nullptr};
    int data_lengths[3] = {0, 0, 0};
    int32_t pixel_strides[3] = {0, 0, 0};
    int32_t row_strides[3] = {0, 0, 0};

    for (int32_t p = 0; p < 3; ++p) {
        // AImage_getPlaneData's last out-param is dataLength (bytes), NOT pixel
        // stride; pixel stride is queried separately below.
        if (AImage_getPlaneData(image, p, &planes[p], &data_lengths[p]) != AMEDIA_OK ||
            planes[p] == nullptr || data_lengths[p] <= 0) {
            return false;
        }
        if (AImage_getPlanePixelStride(image, p, &pixel_strides[p]) != AMEDIA_OK ||
            pixel_strides[p] <= 0) {
            return false;
        }
        if (AImage_getPlaneRowStride(image, p, &row_strides[p]) != AMEDIA_OK ||
            row_strides[p] <= 0) {
            return false;
        }
    }

    const uint8_t * y = planes[0];
    const uint8_t * u = planes[1];
    const uint8_t * v = planes[2];

    for (int32_t j = 0; j < h; ++j) {
        uint8_t * dst = rgb + static_cast<size_t>(j) * static_cast<size_t>(w) * 3;
        const size_t y_row = static_cast<size_t>(j) * static_cast<size_t>(row_strides[0]);
        const size_t cy = static_cast<size_t>(j) / 2;
        const size_t u_row = cy * static_cast<size_t>(row_strides[1]);
        const size_t v_row = cy * static_cast<size_t>(row_strides[2]);
        for (int32_t i = 0; i < w; ++i) {
            const size_t cx = static_cast<size_t>(i) / 2;
            const size_t y_off = y_row + static_cast<size_t>(i) * static_cast<size_t>(pixel_strides[0]);
            const size_t u_off = u_row + cx * static_cast<size_t>(pixel_strides[1]);
            const size_t v_off = v_row + cx * static_cast<size_t>(pixel_strides[2]);

            if (y_off >= static_cast<size_t>(data_lengths[0]) ||
                u_off >= static_cast<size_t>(data_lengths[1]) ||
                v_off >= static_cast<size_t>(data_lengths[2])) {
                return false;  // out-of-bounds sample: abort, never submit a partial frame
            }

            const int Y = y[y_off];
            const int U = u[u_off];
            const int V = v[v_off];

            const int C = Y - 16;
            const int D = U - 128;
            const int E = V - 128;

            const int r = (298 * C + 409 * E + 128) >> 8;
            const int g = (298 * C - 100 * D - 208 * E + 128) >> 8;
            const int b = (298 * C + 516 * D + 128) >> 8;

            dst[i * 3 + 0] = clamp255(r);
            dst[i * 3 + 1] = clamp255(g);
            dst[i * 3 + 2] = clamp255(b);
        }
    }
    return true;
}

} // namespace

struct VideoReader::Impl {
    AMediaExtractor * extractor = nullptr;
    AMediaCodec * codec = nullptr;
    AImageReader * image_reader = nullptr;

    // AImageReader delivers frames on its own thread; ``next()`` blocks on this
    // pair until the listener reports one.
    AImageReader_ImageListener listener{};
    std::mutex mutex;
    std::condition_variable cv;
    int pending_images = 0;

    bool input_eos = false;
    bool output_eos = false;
    bool codec_started = false;  // true only after AMediaCodec_start() succeeds
    int64_t eos_started_at = -1;  // steady_clock ms when codec output EOS was first seen

    int64_t frame_index = 0;
    double fps = 30.0;

    static void on_image_available(void * context, AImageReader *) {
        auto * im = static_cast<Impl *>(context);
        {
            std::lock_guard<std::mutex> lock(im->mutex);
            ++im->pending_images;
        }
        im->cv.notify_one();
    }

    // Push one encoded sample from the extractor into the codec when it has a
    // free input buffer; queue end-of-stream once the extractor is exhausted.
    void feed_input() {
        const ssize_t idx = AMediaCodec_dequeueInputBuffer(codec, 0);
        if (idx < 0) return;

        size_t capacity = 0;
        uint8_t * buf = AMediaCodec_getInputBuffer(codec, static_cast<size_t>(idx), &capacity);
        if (buf == nullptr || capacity == 0) return;

        const ssize_t n = AMediaExtractor_readSampleData(extractor, buf, capacity);
        if (n < 0) {
            AMediaCodec_queueInputBuffer(codec, static_cast<size_t>(idx), 0, 0, 0,
                                         AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
            input_eos = true;
            return;
        }

        const int64_t pts_us = AMediaExtractor_getSampleTime(extractor);
        AMediaCodec_queueInputBuffer(codec, static_cast<size_t>(idx), 0,
                                     static_cast<size_t>(n),
                                     pts_us > 0 ? static_cast<uint64_t>(pts_us) : 0, 0);
        AMediaExtractor_advance(extractor);
    }

    // Render ready output buffers to the AImageReader surface; the listener
    // signals next() once the resulting image is available.
    void drain_output() {
        AMediaCodecBufferInfo info;
        const ssize_t idx = AMediaCodec_dequeueOutputBuffer(codec, &info, 0);
        if (idx < 0) return;  // TRY_AGAIN_LATER / format-change: nothing to render

        const bool render = (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) == 0;
        if (!render) {
            output_eos = true;
            if (eos_started_at < 0) {
                eos_started_at = steady_now_ms();  // start the bounded drain budget
            }
        }
        AMediaCodec_releaseOutputBuffer(codec, static_cast<size_t>(idx), render);
    }

    // Try to take the next decoded frame. Returns false (leaving *out untouched)
    // when none is ready. Only updates pending_images after a successful acquire,
    // so a failed acquire never corrupts the listener's count.
    bool acquire_image(AImage ** out) {
        AImage * image = nullptr;
        if (AImageReader_acquireNextImage(image_reader, &image) != AMEDIA_OK || image == nullptr) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (pending_images > 0) --pending_images;  // never negative
        }
        *out = image;
        return true;
    }

    // Bounded EOS-draining fallback. The NDK exposes no direct signal that the
    // AImageReader surface consumer has fully drained, so after codec EOS we keep
    // retrying for a short budget rather than claiming the last frame cannot be
    // dropped. Actual last-frame behavior must be validated on the target device.
    bool eos_drain_expired() const {
        return eos_started_at >= 0 &&
               steady_now_ms() - eos_started_at >= kEosDrainTimeoutMs;
    }

    // Fills ``out`` from a decoded AImage. Returns false (leaving ``out``
    // unusable) if the dimensions are invalid or the YUV->RGB conversion fails;
    // ``next()`` then skips the frame rather than submitting partial RGB data.
    bool fill_frame(AImage * image, VideoFrame & out) {
        int32_t w = 0;
        int32_t h = 0;
        if (AImage_getWidth(image, &w) != AMEDIA_OK || w <= 0) return false;
        if (AImage_getHeight(image, &h) != AMEDIA_OK || h <= 0) return false;

        out.index = frame_index++;

        int64_t ts_ns = 0;
        if (AImage_getTimestamp(image, &ts_ns) == AMEDIA_OK && ts_ns >= 0) {
            out.timestamp = static_cast<double>(ts_ns) / 1e9;  // ns -> seconds
        } else {
            out.timestamp = static_cast<double>(out.index) / fps;  // fallback
        }

        out.image.width = static_cast<uint32_t>(w);
        out.image.height = static_cast<uint32_t>(h);
        out.image.rgb.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
        return yuv420_to_rgb(image, out.image.rgb.data());
    }

    // Release all native resources and reset to the "not opened" state. Shared
    // by open()'s failure paths and ~Impl(); idempotent. Does not touch err.
    void cleanup() {
        if (codec) {
            if (codec_started) {
                AMediaCodec_stop(codec);  // only valid after start() succeeded
            }
            AMediaCodec_delete(codec);
            codec = nullptr;
        }
        if (image_reader) {
            AImageReader_delete(image_reader);
            image_reader = nullptr;
        }
        if (extractor) {
            AMediaExtractor_delete(extractor);
            extractor = nullptr;
        }
        codec_started = false;
    }

    ~Impl() {
        cleanup();
    }
};

VideoReader::VideoReader(std::string path) : impl_(new Impl()), path_(std::move(path)) {}

VideoReader::~VideoReader() = default;

bool VideoReader::open(std::string & err) {
    Impl & im = *impl_;

    // A reader is single-use: refuse a second open rather than overwriting the
    // first session's resources and leaving the object in an inconsistent state.
    if (opened_) {
        err = "video reader is already open";
        return false;
    }

    im.extractor = AMediaExtractor_new();
    if (im.extractor == nullptr) {
        err = "cannot create AMediaExtractor";
        return false;
    }
    if (AMediaExtractor_setDataSource(im.extractor, path_.c_str()) != AMEDIA_OK) {
        err = "cannot open video: " + path_;
        im.cleanup();
        return false;
    }

    std::string mime;
    size_t track_index = static_cast<size_t>(-1);
    int width = 0;
    int height = 0;
    double fps = 30.0;
    int64_t duration_us = 0;

    const size_t track_count = AMediaExtractor_getTrackCount(im.extractor);
    for (size_t i = 0; i < track_count; ++i) {
        AMediaFormat * fmt = AMediaExtractor_getTrackFormat(im.extractor, i);
        if (fmt == nullptr) {
            continue;  // skip tracks whose format cannot be queried
        }
        //const char * m = AMediaFormat_getString(fmt, AMEDIAFORMAT_KEY_MIME);
        const char *m = nullptr;
        AMediaFormat_getString(fmt, AMEDIAFORMAT_KEY_MIME, &m);
        if (m != nullptr && std::strncmp(m, "video/", 6) == 0) {
            mime = m;
            track_index = i;
            AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_WIDTH, &width);
            AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_HEIGHT, &height);
            AMediaFormat_getInt64(fmt, AMEDIAFORMAT_KEY_DURATION, &duration_us);
            int32_t frame_rate = 0;
            if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_FRAME_RATE, &frame_rate) &&
                frame_rate > 0) {
                fps = static_cast<double>(frame_rate);
            }
            AMediaFormat_delete(fmt);
            break;
        }
        AMediaFormat_delete(fmt);
    }

    if (mime.empty() || track_index == static_cast<size_t>(-1)) {
        err = "no video track: " + path_;
        im.cleanup();
        return false;
    }
    if (width <= 0 || height <= 0) {
        err = "video track reports no dimensions: " + path_;
        im.cleanup();
        return false;
    }
    if (AMediaExtractor_selectTrack(im.extractor, track_index) != AMEDIA_OK) {
        err = "cannot select video track: " + path_;
        im.cleanup();
        return false;
    }

    im.fps = fps;
    duration_ = duration_us > 0 ? static_cast<double>(duration_us) / 1e6 : 0.0;

    // CPU-accessible YUV surface: the codec renders decoded frames into this
    // reader, and next() pulls AImage objects back out for the YUV -> RGB step.
    if (AImageReader_new(width, height, AIMAGE_FORMAT_YUV_420_888, kMaxReaderImages,
                         &im.image_reader) != AMEDIA_OK ||
        im.image_reader == nullptr) {
        err = "cannot create AImageReader: " + path_;
        im.cleanup();
        return false;
    }
    ANativeWindow * window = nullptr;
    if (AImageReader_getWindow(im.image_reader, &window) != AMEDIA_OK || window == nullptr) {
        err = "cannot get ANativeWindow from AImageReader: " + path_;
        im.cleanup();
        return false;
    }

    im.codec = AMediaCodec_createDecoderByType(mime.c_str());
    if (im.codec == nullptr) {
        err = "no decoder for " + mime + ": " + path_;
        im.cleanup();
        return false;
    }

    // Configure with the extractor's track format (carries width/height/csd)
    // and the image reader's window as the output surface.
    AMediaFormat * fmt = AMediaExtractor_getTrackFormat(im.extractor, track_index);
    if (fmt == nullptr) {
        err = "cannot get video track format: " + path_;
        im.cleanup();
        return false;
    }
    const media_status_t cfg = AMediaCodec_configure(im.codec, fmt, window, nullptr, 0);
    AMediaFormat_delete(fmt);
    if (cfg != AMEDIA_OK) {
        err = "cannot configure decoder: " + path_;
        im.cleanup();
        return false;
    }

    im.listener.context = impl_.get();
    im.listener.onImageAvailable = &Impl::on_image_available;
    if (AImageReader_setImageListener(im.image_reader, &im.listener) != AMEDIA_OK) {
        err = "cannot set image listener: " + path_;
        im.cleanup();
        return false;
    }

    if (AMediaCodec_start(im.codec) != AMEDIA_OK) {
        err = "cannot start decoder: " + path_;
        im.cleanup();
        return false;
    }
    im.codec_started = true;  // only now is AMediaCodec_stop() valid

    opened_ = true;
    return true;
}

bool VideoReader::next(VideoFrame & out) {
    if (!opened_) return false;
    Impl & im = *impl_;

    while (true) {
        // Keep the decoder fed and drained; frames arrive via the listener.
        if (!im.input_eos) {
            im.feed_input();
        }
        im.drain_output();

        // Acquire first: AImageReader_acquireNextImage is the source of truth for
        // whether a decoded frame is ready; pending_images is only updated after a
        // successful acquire.
        AImage * image = nullptr;
        if (im.acquire_image(&image)) {
            const bool converted = im.fill_frame(image, out);
            AImage_delete(image);
            if (converted) {
                return true;
            }
            // Conversion failed (invalid plane data): skip this frame rather than
            // submit a partial RGB image, and keep decoding the rest of the stream.
            continue;
        }

        // No frame ready yet. After codec EOS, keep retrying briefly for the last
        // frames (bounded fallback), then stop once the budget expires.
        std::unique_lock<std::mutex> lock(im.mutex);
        if (im.output_eos && im.eos_drain_expired()) {
            return false;
        }
        // Wait for the listener to signal a new image, or poll again shortly to
        // re-feed input / re-drain output.
        im.cv.wait_for(lock, std::chrono::milliseconds(kPollIntervalMs));
    }
}

} // namespace ours
