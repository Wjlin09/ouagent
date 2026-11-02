// Region-of-interest selection.
//
// TODO(region selection): return the salient crop instead of the full frame.

#ifndef OURS_NATIVE_ALGORITHMS_REGION_SELECTOR_H
#define OURS_NATIVE_ALGORITHMS_REGION_SELECTOR_H

#include "common.h"

namespace ours {

class RegionSelector {
public:
    // Identity for now.
    Image select(const Image & frame) const { return frame; }
};

} // namespace ours

#endif // OURS_NATIVE_ALGORITHMS_REGION_SELECTOR_H
