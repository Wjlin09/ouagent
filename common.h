// Shared image/frame types and helpers for the C++ OURS_Agent.


#ifndef OURS_NATIVE_COMMON_H
#define OURS_NATIVE_COMMON_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace ours {

// An owned RGB image: ``width * height * 3`` bytes in ``RGBRGB...`` row-major
// order.
struct Image {
    std::vector<uint8_t> rgb;
    uint32_t width = 0;
    uint32_t height = 0;

    Image() = default;
    Image(std::vector<uint8_t> data, uint32_t w, uint32_t h)
        : rgb(std::move(data)), width(w), height(h) {}

    bool empty() const { return rgb.empty(); }
    size_t n_bytes() const { return static_cast<size_t>(width) * height * 3; }
};

// Wall-clock seconds, replacing Python ``time.time()``.
inline double now_seconds() {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// Aspect-preserving target size: cap the long side at ``max_size`` (never
// upscales). Mirrors ``GemmaBackend._resize_dims``.
inline void resize_dims(uint32_t w, uint32_t h, uint32_t max_size,
                        uint32_t & out_w, uint32_t & out_h) {
    out_w = w;
    out_h = h;
    const uint32_t longest = w > h ? w : h;
    if (longest <= max_size) {
        return;
    }
    const double scale = static_cast<double>(max_size) / static_cast<double>(longest);
    out_w = static_cast<uint32_t>(w * scale + 0.5);
    out_h = static_cast<uint32_t>(h * scale + 0.5);
    if (out_w < 1) out_w = 1;
    if (out_h < 1) out_h = 1;
}

// Bilinear resize of an RGB image. Functional native equivalent of
// ``cv2.resize(..., INTER_AREA)`` -- not bit-identical, but preserves the
// aspect/长边 cap behaviour that drives visual-token count.
inline Image resize_image(const Image & src, uint32_t max_size) {
    uint32_t dw = 0, dh = 0;
    resize_dims(src.width, src.height, max_size, dw, dh);
    if (dw == src.width && dh == src.height) {
        return src;
    }

    Image out;
    out.width = dw;
    out.height = dh;
    out.rgb.resize(static_cast<size_t>(dw) * dh * 3);

    auto clamp = [](int v, uint32_t hi) {
        if (v < 0) return 0;
        if (static_cast<uint32_t>(v) >= hi) return static_cast<int>(hi) - 1;
        return v;
    };

    for (uint32_t y = 0; y < dh; ++y) {
        const double sy = (static_cast<double>(y) + 0.5) * src.height / dh - 0.5;
        const int y0 = static_cast<int>(std::floor(sy));
        const double fy = sy - y0;
        const int cy0 = clamp(y0, src.height);
        const int cy1 = clamp(y0 + 1, src.height);
        for (uint32_t x = 0; x < dw; ++x) {
            const double sx = (static_cast<double>(x) + 0.5) * src.width / dw - 0.5;
            const int x0 = static_cast<int>(std::floor(sx));
            const double fx = sx - x0;
            const int cx0 = clamp(x0, src.width);
            const int cx1 = clamp(x0 + 1, src.width);
            for (int c = 0; c < 3; ++c) {
                const double v00 = src.rgb[(cy0 * src.width + cx0) * 3 + c];
                const double v01 = src.rgb[(cy0 * src.width + cx1) * 3 + c];
                const double v10 = src.rgb[(cy1 * src.width + cx0) * 3 + c];
                const double v11 = src.rgb[(cy1 * src.width + cx1) * 3 + c];
                const double top = v00 * (1.0 - fx) + v01 * fx;
                const double bot = v10 * (1.0 - fx) + v11 * fx;
                const double v = top * (1.0 - fy) + bot * fy;
                out.rgb[(y * dw + x) * 3 + c] = static_cast<uint8_t>(v + 0.5);
            }
        }
    }
    return out;
}

} // namespace ours

#endif // OURS_NATIVE_COMMON_H
