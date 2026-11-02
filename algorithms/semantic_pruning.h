// Query-aware semantic pruning interface. 
// Not completed yet.

#ifndef OURS_NATIVE_ALGORITHMS_SEMANTIC_PRUNING_H
#define OURS_NATIVE_ALGORITHMS_SEMANTIC_PRUNING_H

#include "common.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace ours {

class SemanticPruner {
public:
    virtual ~SemanticPruner() = default;

    // ``query`` is empty before the first start_query (query-agnostic) and the
    // current question afterwards (query-guided). ``session_state`` is opaque
    // session context (AgentState) a real scorer may need.
    virtual std::optional<Image> select(const Image & frame, const std::string & query,
                                        const void * session_state) {
        (void) frame;
        (void) query;
        (void) session_state;
        throw std::runtime_error(
            "no concrete query-aware semantic pruner is implemented; "
            "enable algorithms.semantic_pruning only after providing a real "
            "scorer (an identity pass-through is not a semantic algorithm)");
    }
};

} // namespace ours

#endif // OURS_NATIVE_ALGORITHMS_SEMANTIC_PRUNING_H
