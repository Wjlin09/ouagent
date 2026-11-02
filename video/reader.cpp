// VideoReader implementation (ffmpeg/libav decode to RGB24).
//
// NOTE: this file requires the ffmpeg development headers/libs (libavformat,
// libavcodec, libswscale, libavutil) -- the same system dependency llama.cpp's
// MTMD_VIDEO build uses on the inference server. It is guarded by
// ``OURS_HAVE_FFMPEG`` (set by CMake when pkg-config finds ffmpeg) so the C++
// Agent still builds (with a no-decode stub) on machines without ffmpeg dev.

#include "video/reader.h"

#include <cstdio>
#include <utility>

#ifdef OURS_HAVE_FFMPEG
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}
#endif

namespace ours {

#ifdef OURS_HAVE_FFMPEG

struct VideoReader::Impl {
    AVFormatContext * fmt = nullptr;
    AVCodecContext * dec = nullptr;
    AVFrame * frame = nullptr;
    AVFrame * rgb_frame = nullptr;
    SwsContext * sws = nullptr;
    AVPacket * packet = nullptr;
    int stream_index = -1;
    double fps = 30.0;
    int64_t frame_index = 0;
    uint8_t * rgb_buf = nullptr;
    int rgb_buf_size = 0;
    int width = 0;
    int height = 0;

    ~Impl() {
        if (rgb_buf) av_free(rgb_buf);
        if (rgb_frame) av_frame_free(&rgb_frame);
        if (frame) av_frame_free(&frame);
        if (sws) sws_freeContext(sws);
        if (dec) avcodec_free_context(&dec);
        if (fmt) avformat_close_input(&fmt);
        if (packet) av_packet_free(&packet);
    }
};

VideoReader::VideoReader(std::string path) : impl_(new Impl()), path_(std::move(path)) {}

VideoReader::~VideoReader() = default;

bool VideoReader::open(std::string & err) {
    Impl & im = *impl_;
    if (avformat_open_input(&im.fmt, path_.c_str(), nullptr, nullptr) < 0) {
        err = "cannot open video: " + path_;
        return false;
    }
    if (avformat_find_stream_info(im.fmt, nullptr) < 0) {
        err = "cannot find stream info: " + path_;
        return false;
    }

    const AVCodec * codec = nullptr;
    for (unsigned i = 0; i < im.fmt->nb_streams; ++i) {
        const AVStream * st = im.fmt->streams[i];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            im.stream_index = static_cast<int>(i);
            codec = avcodec_find_decoder(st->codecpar->codec_id);
            if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0) {
                im.fps = av_q2d(st->avg_frame_rate);
            }
            break;
        }
    }
    if (im.stream_index < 0 || codec == nullptr) {
        err = "no video stream: " + path_;
        return false;
    }

    im.dec = avcodec_alloc_context3(codec);
    if (im.dec == nullptr) {
        err = "cannot allocate codec context: " + path_;
        return false;
    }
    if (avcodec_parameters_to_context(im.dec, im.fmt->streams[im.stream_index]->codecpar) < 0) {
        err = "cannot copy codec params: " + path_;
        return false;
    }
    if (avcodec_open2(im.dec, codec, nullptr) < 0) {
        err = "cannot open codec: " + path_;
        return false;
    }

    im.frame = av_frame_alloc();
    im.rgb_frame = av_frame_alloc();
    im.packet = av_packet_alloc();
    im.width = im.dec->width;
    im.height = im.dec->height;

    // Container/stream duration in seconds (used by the StreamingBench driver's
    // ``video_duration`` summary). Prefer the container duration; fall back to
    // the video stream's own duration when the container omits it.
    duration_ = 0.0;
    if (im.fmt->duration != AV_NOPTS_VALUE) {
        duration_ = static_cast<double>(im.fmt->duration) / AV_TIME_BASE;
    }
    if (duration_ <= 0.0) {
        const AVStream * st = im.fmt->streams[im.stream_index];
        if (st->duration != AV_NOPTS_VALUE && st->time_base.num > 0 && st->time_base.den > 0) {
            duration_ = static_cast<double>(st->duration) * av_q2d(st->time_base);
        }
    }

    opened_ = true;
    return true;
}

bool VideoReader::next(VideoFrame & out) {
    if (!opened_) return false;
    Impl & im = *impl_;
    bool flushing = false;

    while (true) {
        const int r = avcodec_receive_frame(im.dec, im.frame);
        if (r == 0) {
            // Decoded a frame -> convert to packed RGB24.
            const int w = im.frame->width;
            const int h = im.frame->height;
            if (im.sws == nullptr || w != im.width || h != im.height) {
                if (im.sws) sws_freeContext(im.sws);
                im.sws = sws_getContext(w, h, static_cast<AVPixelFormat>(im.frame->format),
                                       w, h, AV_PIX_FMT_RGB24, SWS_BILINEAR,
                                       nullptr, nullptr, nullptr);
                im.width = w;
                im.height = h;
                const int buf_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, w, h, 1);
                if (buf_size > im.rgb_buf_size) {
                    if (im.rgb_buf) av_free(im.rgb_buf);
                    im.rgb_buf = static_cast<uint8_t *>(av_malloc(buf_size));
                    im.rgb_buf_size = buf_size;
                }
            }

            av_image_fill_arrays(im.rgb_frame->data, im.rgb_frame->linesize,
                                 im.rgb_buf, AV_PIX_FMT_RGB24, w, h, 1);
            sws_scale(im.sws, im.frame->data, im.frame->linesize, 0, h,
                      im.rgb_frame->data, im.rgb_frame->linesize);

            out.index = im.frame_index++;
            //const AVStream * st = im.fmt->streams[im.stream_index];

            //if (im.frame->pts != AV_NOPTS_VALUE &&
            //    st->time_base.num > 0 &&
            //    st->time_base.den > 0) {
            //    out.timestamp = im.frame->pts * av_q2d(st->time_base);
            //    out.timestamp = im.frame->pts * av_q2d(im.fmt->streams[im.stream_index]->time_base);
            //    out.timestamp = static_cast<double>(out.index) / im.fps;

            //} else {
            //    out.timestamp = static_cast<double>(out.index) / im.fps;
            //}
            out.timestamp = static_cast<double>(out.index) / im.fps;
            out.image.width = static_cast<uint32_t>(w);
            out.image.height = static_cast<uint32_t>(h);
            out.image.rgb.assign(im.rgb_buf, im.rgb_buf + static_cast<size_t>(w) * h * 3);
            return true;
        }

        if (r == AVERROR_EOF) {
            return false;
        }

        // r == AVERROR(EAGAIN): the decoder needs more input.
        if (flushing) {
            return false;  // already flushed, nothing left
        }
        const int pr = av_read_frame(im.fmt, im.packet);
        if (pr < 0) {
            // End of file: flush buffered frames out of the decoder.
            avcodec_send_packet(im.dec, nullptr);
            flushing = true;
            continue;
        }
        if (im.packet->stream_index == im.stream_index) {
            avcodec_send_packet(im.dec, im.packet);
        }
        av_packet_unref(im.packet);
    }
}

#else  // !OURS_HAVE_FFMPEG

struct VideoReader::Impl {
    int unused = 0;
};

VideoReader::VideoReader(std::string path) : impl_(new Impl()), path_(std::move(path)) {}

VideoReader::~VideoReader() = default;

bool VideoReader::open(std::string & err) {
    err = "VideoReader was built without ffmpeg (OURS_HAVE_FFMPEG undefined). "
          "Install the ffmpeg dev libraries (already needed by llama.cpp's "
          "MTMD_VIDEO build) and reconfigure.";
    return false;
}

bool VideoReader::next(VideoFrame & out) {
    (void) out;
    return false;
}

#endif  // OURS_HAVE_FFMPEG

} // namespace ours
