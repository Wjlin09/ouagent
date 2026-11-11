// Session-level chunk history for OURS_Agent (Cpp-GPU).
//
// One entry per processed video chunk: a mean-pooled visual embedding (for
// chunk-level similarity retrieval) plus the LOGICAL LLM token positions its
// visual tokens occupy in the KV cache (sequence 0).
//
// Important distinctions:
//   * chunk_embedding is the mmproj output in LLM input space (BEFORE LLM
//     prefill). It is NOT a KV vector, and the buffer never stores per-frame
//     embeddings or raw KV data -- only the pooled chunk embedding + metadata.
//   * visual_positions stores logical llama_pos ranges, NOT physical KV-cache
//     memory addresses.
//   * Visual token positions are NOT assumed contiguous across frames
//     (interleaved text tokens also consume positions). `position_mapping_valid`
//     is true only when every media chunk maps 1:1 (n_pos == n_tokens, i.e.
//     non-M-RoPE). When false the range mapping is unreliable and the entry
//     must NOT be used for retrieval.
//
// Retrieval gate: encoded && prefill_completed && position_mapping_valid.

#ifndef OURS_NATIVE_CHUNK_HISTORY_H
#define OURS_NATIVE_CHUNK_HISTORY_H

#include <cstdint>
#include <vector>

namespace ours {

// A contiguous run of logical token positions occupied by one frame's visual
// tokens: [start, start + count).
struct VisualTokenRange {
    int64_t start = 0;
    int64_t count = 0;
};

struct ChunkHistoryEntry {
    int64_t chunk_id = -1;
    std::vector<float> chunk_embedding;  // mean-pooled mmproj output (LLM input space)
    int64_t embedding_dim = 0;           // == llama_model_n_embd_inp(model)
    int64_t visual_token_count = 0;      // total visual tokens across frames
    std::vector<VisualTokenRange> visual_positions;  // valid iff position_mapping_valid
    bool encoded = false;                // Stage A: ViT + mmproj encode completed
    bool prefill_completed = false;      // Stage B: all visual embeddings prefilled
    bool position_mapping_valid = false; // every media chunk has n_pos == n_tokens
};

} // namespace ours

#endif // OURS_NATIVE_CHUNK_HISTORY_H
