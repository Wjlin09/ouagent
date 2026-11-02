// Frame-level input filtering. 
// Not completed yet

#ifndef OURS_NATIVE_ALGORITHMS_FRAME_SELECTOR_H
#define OURS_NATIVE_ALGORITHMS_FRAME_SELECTOR_H

#include "common.h"

#include <optional>

namespace ours {

class FrameSelector {
public:
    virtual ~FrameSelector() = default;

    // Return std::nullopt to drop the frame; otherwise return the frame to
    // keep (which may differ from the input).
    virtual std::optional<Image> select(const Image & frame) = 0;
};

} // namespace ours

#endif // OURS_NATIVE_ALGORITHMS_FRAME_SELECTOR_H
