// Token pruning strategy. 
// Not completed yet.

#ifndef OURS_NATIVE_ALGORITHMS_TOKEN_SELECTOR_H
#define OURS_NATIVE_ALGORITHMS_TOKEN_SELECTOR_H

#include <vector>

namespace ours {

class TokenSelector {
public:
    // Identity no-op placeholder. Do NOT treat as real token pruning.
    template <typename T>
    std::vector<T> prune(const std::vector<T> & tokens) const {
        return tokens;
    }
};

} // namespace ours

#endif // OURS_NATIVE_ALGORITHMS_TOKEN_SELECTOR_H
