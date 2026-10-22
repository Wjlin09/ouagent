"""ctypes binding to llama.cpp's ``libmtmd`` + ``libllama`` C API.

This is the *only* module in OURS_Agent that talks to native code. Everything
above it (``GemmaBackend``, ``Agent``) works with plain Python objects
(``str`` / ``bytes`` / ``int``) and never sees a ctypes pointer or a C struct.

The API surface here is transcribed from the llama.cpp source vendored at
``third_party/llama.cpp-master``:

    * ``tools/mtmd/mtmd.h``         -- multimodal projector / mmproj C API
    * ``tools/mtmd/mtmd-helper.h``  -- high-level eval helpers (same library)
    * ``include/llama.h``           -- text model / decode / sampling C API

"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path: Optional[str]) -> Optional[str]:
    """Resolve ``path`` to an absolute path, relative to the project root."""
    if not path:
        return None
    p = Path(path)
    return str(p if p.is_absolute() else (_PROJECT_ROOT / p))


class LibmtmdError(RuntimeError):
    """Raised when the native runtime is missing or a native call fails."""


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _shared_library_names(name: str) -> list[str]:
    """Return candidate file names for a shared library across platforms."""
    if sys.platform == "win32":
        return [f"{name}.dll"]
    if sys.platform == "darwin":
        return [f"lib{name}.dylib", f"lib{name}.0.dylib"]
    # Linux and other POSIX systems.
    return [f"lib{name}.so", f"lib{name}.so.0", f"lib{name}.so.1"]


def _load_library(
    display_name: str,
    env_var: str,
    name: str,
    mode: Optional[int] = None,
) -> ctypes.CDLL:
    """Load a shared library, honouring an explicit path env var first.

    ``mode`` is passed to :func:`ctypes.CDLL` (e.g. ``ctypes.RTLD_GLOBAL`` on
    POSIX) so dependent libraries can resolve symbols against this one.

    Raises :class:`LibmtmdError` with a clear message when the library cannot
    be found, so a host without libmtmd/libllama fails only at *load* time
    (i.e. on the first ``infer()``), never at ``import`` time.
    """
    candidates: list[str] = []

    explicit = os.environ.get(env_var)
    if explicit:
        candidates.append(explicit)

    found = ctypes.util.find_library(name)
    if found:
        candidates.append(found)

    candidates.extend(_shared_library_names(name))

    seen: set[str] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            if mode is not None:
                return ctypes.CDLL(path, mode=mode)
            return ctypes.CDLL(path)
        except OSError as exc:
            logger.debug("failed to load %s from %s: %s", display_name, path, exc)

    raise LibmtmdError(
        f"{display_name} is not available. "
        f"Please configure llama.cpp/libmtmd on the inference server "
        f"(searched: {', '.join(seen) or '<nothing>'}). "
        f"Set {env_var} to an explicit path to the built library."
    )


# ---------------------------------------------------------------------------
# ctypes struct definitions (transcribed from the vendored headers).
# ---------------------------------------------------------------------------

class MtmdContextParams(ctypes.Structure):
    """``struct mtmd_context_params`` from ``mtmd.h``."""

    _fields_ = [
        ("use_gpu", ctypes.c_bool),
        ("device", ctypes.c_void_p),                     # ggml_backend_dev_t
        ("print_timings", ctypes.c_bool),
        ("n_threads", ctypes.c_int),
        ("image_marker", ctypes.c_char_p),               # deprecated
        ("media_marker", ctypes.c_char_p),
        ("flash_attn_type", ctypes.c_int),               # enum llama_flash_attn_type
        ("warmup", ctypes.c_bool),
        ("image_min_tokens", ctypes.c_int),
        ("image_max_tokens", ctypes.c_int),
        ("cb_eval", ctypes.c_void_p),                    # ggml_backend_sched_eval_callback
        ("cb_eval_user_data", ctypes.c_void_p),
        ("batch_max_tokens", ctypes.c_int32),
        ("progress_callback", ctypes.c_void_p),          # mtmd_progress_callback
        ("progress_callback_user_data", ctypes.c_void_p),
    ]


class MtmdInputText(ctypes.Structure):
    """``struct mtmd_input_text`` from ``mtmd.h``."""

    _fields_ = [
        ("text", ctypes.c_char_p),
        ("text_len", ctypes.c_size_t),
        ("add_special", ctypes.c_bool),
        ("parse_special", ctypes.c_bool),
    ]


class MtmdInputPart(ctypes.Structure):
    """``struct mtmd_input_part`` from ``mtmd.h`` (one text or bitmap, not both)."""

    _fields_ = [
        ("text", ctypes.POINTER(MtmdInputText)),
        ("bitmap", ctypes.c_void_p),                    # const mtmd_bitmap *
    ]


class LlamaModelParams(ctypes.Structure):
    """``struct llama_model_params`` from ``llama.h`` (fields kept in order)."""

    _fields_ = [
        ("devices", ctypes.c_void_p),                    # ggml_backend_dev_t *
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),                    # enum llama_split_mode
        ("load_mode", ctypes.c_int),                     # enum llama_load_mode
        ("lazy_mode", ctypes.c_int),                     # enum llama_lazy_mode
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),               # const float *
        ("progress_callback", ctypes.c_void_p),          # llama_progress_callback
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class LlamaContextParams(ctypes.Structure):
    """``struct llama_context_params`` from ``llama.h`` (fields kept in order)."""

    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int),                      # enum llama_context_type
        ("rope_scaling_type", ctypes.c_int),             # enum llama_rope_scaling_type
        ("pooling_type", ctypes.c_int),                  # enum llama_pooling_type
        ("attention_type", ctypes.c_int),                # enum llama_attention_type
        ("flash_attn_type", ctypes.c_int),               # enum llama_flash_attn_type
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),                    # ggml_backend_sched_eval_callback
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),                        # enum ggml_type
        ("type_v", ctypes.c_int),                        # enum ggml_type
        ("moe_cache_size", ctypes.c_size_t),
        ("abort_callback", ctypes.c_void_p),             # ggml_abort_callback
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),                   # llama_sampler_seq_config *
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),                  # llama_context *
    ]


class LlamaSamplerChainParams(ctypes.Structure):
    """``struct llama_sampler_chain_params`` from ``llama.h``."""

    _fields_ = [
        ("no_perf", ctypes.c_bool),
    ]


class LlamaBatch(ctypes.Structure):
    """``struct llama_batch`` from ``llama.h``."""

    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(ctypes.c_int32)),       # llama_token *
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(ctypes.c_int32)),         # llama_pos *
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),  # llama_seq_id **
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


# ---------------------------------------------------------------------------
# Function binding. ``_configure_*`` set argtypes/restype on the loaded CDLL.
# ---------------------------------------------------------------------------

_GGML_LOG_LEVEL_WARN = 3
_GGML_LOG_LEVEL_ERROR = 4


def _filtered_log_callback(level: int, text, user_data) -> None:
    """Drop ggml debug/info logs; keep warnings and errors on stderr."""
    if level in (_GGML_LOG_LEVEL_WARN, _GGML_LOG_LEVEL_ERROR):
        msg = text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text)
        sys.stderr.write(msg)
        sys.stderr.flush()


# C signature: void (*ggml_log_callback)(enum ggml_log_level, const char *, void *)
_GGML_LOG_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)
# Module-level reference so the callback is never garbage-collected while the
# native library still holds a pointer to it.
_LOG_FILTER = _GGML_LOG_CALLBACK(_filtered_log_callback)


def _configure_mtmd(lib: ctypes.CDLL) -> None:
    """Bind the ``mtmd.h`` + ``mtmd-helper.h`` C API (all in libmtmd)."""
    f = lib

    f.mtmd_default_marker.restype = ctypes.c_char_p

    f.mtmd_context_params_default.restype = MtmdContextParams

    f.mtmd_init_from_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p, MtmdContextParams]
    f.mtmd_init_from_file.restype = ctypes.c_void_p
    f.mtmd_free.argtypes = [ctypes.c_void_p]

    f.mtmd_support_vision.argtypes = [ctypes.c_void_p]
    f.mtmd_support_vision.restype = ctypes.c_bool

    f.mtmd_bitmap_init.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    f.mtmd_bitmap_init.restype = ctypes.c_void_p
    f.mtmd_bitmap_free.argtypes = [ctypes.c_void_p]

    f.mtmd_input_chunks_init.restype = ctypes.c_void_p
    f.mtmd_input_chunks_free.argtypes = [ctypes.c_void_p]

    f.mtmd_tokenize.argtypes = [
        ctypes.c_void_p,  # const mtmd_context *
        ctypes.c_void_p,  # mtmd_input_chunks * (output)
        ctypes.POINTER(MtmdInputText),  # const mtmd_input_text *
        ctypes.POINTER(ctypes.c_void_p),  # const mtmd_bitmap * const *
        ctypes.c_size_t,  # n_bitmaps
    ]
    f.mtmd_tokenize.restype = ctypes.c_int32

    # Same tokenizer as mtmd_tokenize(), but takes explicit text/bitmap parts
    # (the CLI path). Per-part add_special is ignored; the trailing add_special
    # is what requests the single BOS token.
    f.mtmd_tokenize_from_parts.argtypes = [
        ctypes.c_void_p,  # const mtmd_context *
        ctypes.c_void_p,  # mtmd_input_chunks * (output)
        ctypes.POINTER(ctypes.POINTER(MtmdInputPart)),  # const mtmd_input_part * const *
        ctypes.c_size_t,  # n_parts
        ctypes.c_bool,    # add_special
    ]
    f.mtmd_tokenize_from_parts.restype = ctypes.c_int32

    # mtmd-helper.h (same library). This decodes text chunks and, for
    # image/audio chunks, runs mtmd_encode_chunk() + decode internally
    # (see mtmd-helper.cpp: mtmd_helper_eval_chunks).
    f.mtmd_helper_eval_chunks.argtypes = [
        ctypes.c_void_p,  # mtmd_context *
        ctypes.c_void_p,  # llama_context *
        ctypes.c_void_p,  # const mtmd_input_chunks *
        ctypes.c_int32,   # llama_pos n_past
        ctypes.c_int32,   # llama_seq_id seq_id
        ctypes.c_int32,   # int32_t n_batch
        ctypes.c_bool,    # bool logits_last
        ctypes.POINTER(ctypes.c_int32),  # llama_pos * new_n_past (output)
    ]
    f.mtmd_helper_eval_chunks.restype = ctypes.c_int32

    f.mtmd_helper_get_n_pos.argtypes = [ctypes.c_void_p]  # const mtmd_input_chunks *
    f.mtmd_helper_get_n_pos.restype = ctypes.c_int32      # llama_pos

    # Diagnostic accessors: read back the tokenized text tokens so the last
    # context token ID can be logged before sampling. Read-only.
    f.mtmd_input_chunks_size.argtypes = [ctypes.c_void_p]
    f.mtmd_input_chunks_size.restype = ctypes.c_size_t
    f.mtmd_input_chunks_get.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    f.mtmd_input_chunks_get.restype = ctypes.c_void_p
    f.mtmd_input_chunk_get_tokens_text.argtypes = [
        ctypes.c_void_p,  # const mtmd_input_chunk *
        ctypes.POINTER(ctypes.c_size_t),  # size_t * n_tokens_output
    ]
    f.mtmd_input_chunk_get_tokens_text.restype = ctypes.POINTER(ctypes.c_int32)
    f.mtmd_input_chunk_get_type.argtypes = [ctypes.c_void_p]
    f.mtmd_input_chunk_get_type.restype = ctypes.c_int32  # enum mtmd_input_chunk_type
    f.mtmd_input_chunk_get_n_tokens.argtypes = [ctypes.c_void_p]
    f.mtmd_input_chunk_get_n_tokens.restype = ctypes.c_size_t

    # Log-callback control: swap the default "print every level to stderr" for
    # the level-filtered callback (drops debug/info, keeps warn/error).
    f.mtmd_log_set.argtypes = [_GGML_LOG_CALLBACK, ctypes.c_void_p]
    if hasattr(f, "mtmd_helper_log_set"):
        f.mtmd_helper_log_set.argtypes = [_GGML_LOG_CALLBACK, ctypes.c_void_p]


def _configure_llama(lib: ctypes.CDLL) -> None:
    """Bind the subset of the ``llama.h`` C API needed for inference."""
    f = lib

    f.llama_backend_init.argtypes = []
    f.llama_log_set.argtypes = [_GGML_LOG_CALLBACK, ctypes.c_void_p]

    f.llama_model_default_params.restype = LlamaModelParams
    f.llama_model_load_from_file.argtypes = [ctypes.c_char_p, LlamaModelParams]
    f.llama_model_load_from_file.restype = ctypes.c_void_p
    f.llama_model_free.argtypes = [ctypes.c_void_p]

    f.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    f.llama_model_get_vocab.restype = ctypes.c_void_p

    f.llama_model_chat_template.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    f.llama_model_chat_template.restype = ctypes.c_char_p

    f.llama_context_default_params.restype = LlamaContextParams
    f.llama_init_from_model.argtypes = [ctypes.c_void_p, LlamaContextParams]
    f.llama_init_from_model.restype = ctypes.c_void_p
    f.llama_free.argtypes = [ctypes.c_void_p]

    f.llama_n_batch.argtypes = [ctypes.c_void_p]
    f.llama_n_batch.restype = ctypes.c_uint32

    f.llama_get_memory.argtypes = [ctypes.c_void_p]
    f.llama_get_memory.restype = ctypes.c_void_p  # llama_memory_t
    f.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]

    f.llama_token_to_piece.argtypes = [
        ctypes.c_void_p,  # const llama_vocab *
        ctypes.c_int32,   # llama_token
        ctypes.c_char_p,  # char * buf
        ctypes.c_int32,   # int32_t length
        ctypes.c_int32,   # int32_t lstrip
        ctypes.c_bool,    # bool special
    ]
    f.llama_token_to_piece.restype = ctypes.c_int32

    f.llama_vocab_is_eog.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    f.llama_vocab_is_eog.restype = ctypes.c_bool
    f.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    f.llama_vocab_n_tokens.restype = ctypes.c_int32
    f.llama_vocab_eot.argtypes = [ctypes.c_void_p]
    f.llama_vocab_eot.restype = ctypes.c_int32  # llama_token (end-of-turn)

    f.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    f.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

    f.llama_decode.argtypes = [ctypes.c_void_p, LlamaBatch]
    f.llama_decode.restype = ctypes.c_int32

    f.llama_sampler_chain_default_params.restype = LlamaSamplerChainParams
    f.llama_sampler_chain_init.argtypes = [LlamaSamplerChainParams]
    f.llama_sampler_chain_init.restype = ctypes.c_void_p
    f.llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    f.llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
    f.llama_sampler_init_dist.restype = ctypes.c_void_p
    f.llama_sampler_init_temp.argtypes = [ctypes.c_float]
    f.llama_sampler_init_temp.restype = ctypes.c_void_p
    f.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    f.llama_sampler_init_top_p.restype = ctypes.c_void_p
    f.llama_sampler_init_penalties.argtypes = [
        ctypes.c_int32,  # n_vocab
        ctypes.c_int32,  # penalty_last_n
        ctypes.c_float,  # penalty_repeat
        ctypes.c_float,  # penalty_freq
        ctypes.c_float,  # penalty_present
    ]
    f.llama_sampler_init_penalties.restype = ctypes.c_void_p
    f.llama_sampler_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    f.llama_sampler_sample.restype = ctypes.c_int32
    f.llama_sampler_accept.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    f.llama_sampler_reset.argtypes = [ctypes.c_void_p]  # void llama_sampler_reset(smpl)
    f.llama_sampler_free.argtypes = [ctypes.c_void_p]


# End-of-turn markers that the native runtime treats as end-of-generation but
# which can reach ``_token_to_piece`` as plain text (spread across one or more
# tokens) when the vocab/lib does not mark them EOG/control. They are matched
# against the *accumulated* generated text, not individual pieces, because a
# marker such as "<end_of_turn>" can be split across several sampled tokens.
_EOT_TEXT_MARKERS = (
    "<end_of_turn>",
    "<turn|>",  # Gemma 4 uses "<turn|>" instead of "<end_of_turn>"
    "<eos>",
    "<|end_of_turn|>",
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_utterance>",
)

# Gemma 4 chat format, transcribed from the vendored llama.cpp template source
#   third_party/llama.cpp-master/models/templates/google-gemma-4-31B-it.jinja
# and common/parsers/gemma4.cpp (common_chat_params_init_gemma4). For a single
# user turn with no system prompt and no tools, llama-mtmd-cli renders the
# template with enable_thinking=true (the common/ layer DEFAULT), which emits a
# leading "<|think|>" system turn and leaves the model's thought channel OPEN:
#
#   <|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{content}<turn|>\n<|turn>model\n
#
# The template's leading <bos> is stripped by the common layer (add_bos=true)
# and re-added by mtmd's add_special, so it is not part of these constants. The
# model then emits its own "<|channel>thought\n...<channel|>" block before the
# answer.
#
# These delimiters are the Gemma 4 format, NOT the legacy Gemma
# <start_of_turn>/<end_of_turn> format. The legacy llama_chat_apply_template()
# exported by libllama.so has no Gemma 4 entry, and the real Gemma 4 Jinja
# engine lives in the common/ layer (not exported by libllama.so), so the
# format is applied here directly rather than via a template-apply API.
#_GEMMA4_USER_OPEN = "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n"
_GEMMA4_USER_OPEN = "<|turn>system\n<turn|>\n<|turn>user\n"
_GEMMA4_USER_CLOSE = "<turn|>\n"
_GEMMA4_MODEL_OPEN = "<|turn>model\n"
_GEMMA4_THINK_SUFFIX = ""  # enable_thinking=true: thought channel is left open
# Official Gemma 4 end-of-turn marker: the template closes *every* message
# (including a model turn cut short by stop_query()) with "<turn|>\n". When a
# query is stopped mid-generation the model turn is left unterminated in the
# KV; the next turn-boundary operation appends exactly this marker to close it,
# mirroring how the reference (mtmd-cli) re-renders the full history. This is
# the reference's own delimiter, not a new special token.
_GEMMA4_TURN_CLOSE = "<turn|>\n"


class NativeMTMDBackend:
    """Low-level backend: Python -> ctypes -> libmtmd + libllama.

    This class is the seam between pure-Python OURS_Agent code and the native
    llama.cpp runtime. 

    TODO(server): integration test required against the real libmtmd/libllama.
    """

    def __init__(
        self,
        *,
        model_path: str,
        mmproj_path: str,
        n_ctx: int = 8192,
        n_batch: int = 2048,
        n_ubatch: int = 512,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        mmproj_use_gpu: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        seed: int = 1234,
        repeat_penalty: float = 1.1,
        verbose: bool = False,
        libllama_path: Optional[str] = None,
        libmtmd_path: Optional[str] = None,
        lib_dir: Optional[str] = None,
    ) -> None:
        self._model_path = model_path
        self._mmproj_path = mmproj_path
        self._n_ctx = int(n_ctx)
        self._n_batch = int(n_batch)
        self._n_ubatch = int(n_ubatch)
        self._n_gpu_layers = int(n_gpu_layers)
        self._n_threads = int(n_threads) if n_threads is not None else None
        self._mmproj_use_gpu = bool(mmproj_use_gpu)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._max_tokens = int(max_tokens)
        self._seed = int(seed)
        self._repeat_penalty = float(repeat_penalty)
        self._verbose = bool(verbose)

        self._libllama_path = libllama_path
        self._libmtmd_path = libmtmd_path
        self._lib_dir = lib_dir

        self._loaded = False
        self._mtmd: Optional[ctypes.CDLL] = None
        self._llama: Optional[ctypes.CDLL] = None
        self._model = None  # llama_model *
        self._lctx = None   # llama_context *
        self._vocab = None  # llama_vocab *
        self._mtmd_ctx = None  # mtmd_context *
        self._smpl = None   # llama_sampler *
        self._media_marker: bytes = b""
        # Benchmark metrics sink (duck-typed ``record_append(...)``), attached by
        # the benchmark driver via :meth:`set_metrics_recorder`. None = no
        # collection. Read-only observation: never changes inference behaviour.
        self._metrics_recorder = None

        # Persistent streaming state: the KV-cache position and the Gemma 4 turn
        # structure are kept across append_video_chunk()/start_query()/
        # step_generation() calls, so one video session is a single continuous
        # context that survives many user queries. These flags are cleared only
        # by start_session()/end_session(); stop_query() never touches them.
        #   ``_turn_active``     : a "<|turn>user\n" turn is open (frames may be
        #                          appended to it).
        #   ``_model_turn_open`` : a "<|turn>model\n" turn is open (generation
        #                          in progress, or cut short by stop_query()).
        #   ``_query_pieces``    : raw text pieces sampled for the active query.
        #   ``_query_tokens``    : number of tokens sampled for the active query
        #                          (per-query ``max_tokens`` cap, reset by
        #                          start_query(); NOT a session-level limit).
        self._n_past = 0
        self._session_active = False
        self._query_active = False
        self._turn_active = False
        self._model_turn_open = False
        self._query_pieces: list[str] = []
        self._query_tokens = 0
        self._n_frames_added = 0
        self._generation_count = 0
        self._logits_logged = False
        self._last_context_token_id = None

    # -- runtime path resolution --------------------------------------------

    def _configure_runtime_paths(self) -> None:
        """Export configured lib paths as env vars (env vars take priority)."""
        if self._libllama_path:
            resolved = _resolve_path(self._libllama_path)
            if os.environ.get("LIBLLAMA_PATH"):
                logger.info(
                    "libllama path from environment (LIBLLAMA_PATH): %s",
                    os.environ["LIBLLAMA_PATH"],
                )
            else:
                os.environ.setdefault("LIBLLAMA_PATH", resolved)
                logger.info("libllama path from runtime config: %s", resolved)

        if self._libmtmd_path:
            resolved = _resolve_path(self._libmtmd_path)
            if os.environ.get("LIBMTMD_PATH"):
                logger.info(
                    "libmtmd path from environment (LIBMTMD_PATH): %s",
                    os.environ["LIBMTMD_PATH"],
                )
            else:
                os.environ.setdefault("LIBMTMD_PATH", resolved)
                logger.info("libmtmd path from runtime config: %s", resolved)

        if self._lib_dir:
            resolved = _resolve_path(self._lib_dir)
            self._prepend_ld_library_path(resolved)
            logger.info("lib_dir from runtime config: %s", resolved)

    @staticmethod
    def _prepend_ld_library_path(dir_path: str) -> None:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in existing.split(os.pathsep) if p]
        if dir_path in parts:
            return
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([dir_path] + parts)

    # -- lifecycle ----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        # Resolve and export configured library paths (env vars take priority).
        self._configure_runtime_paths()

        # Load the native runtime first so a missing libmtmd/libllama surfaces a
        # clear "native runtime is not available" error before any model-file
        # check. libmtmd links against libllama, so libllama is loaded first and
        # its symbols exported globally (RTLD_GLOBAL) so libmtmd resolves against
        # this same libllama instance.
        rtld_global = getattr(ctypes, "RTLD_GLOBAL", None)
        self._llama = _load_library("libllama", "LIBLLAMA_PATH", "llama", mode=rtld_global)
        self._mtmd = _load_library("libmtmd", "LIBMTMD_PATH", "mtmd", mode=rtld_global)
        _configure_mtmd(self._mtmd)
        _configure_llama(self._llama)
        self._install_log_filters()

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"model file not found: {self._model_path}. "
                f"Download the Gemma 4 GGUF on the inference server."
            )
        if not os.path.exists(self._mmproj_path):
            raise FileNotFoundError(
                f"mmproj file not found: {self._mmproj_path}. "
                f"Download the mmproj GGUF on the inference server."
            )

        llama = self._llama
        mtmd = self._mtmd

        llama.llama_backend_init()

        # -- text model ------------------------------------------------------
        mparams = llama.llama_model_default_params()
        mparams.n_gpu_layers = self._n_gpu_layers
        self._model = llama.llama_model_load_from_file(
            os.fsencode(self._model_path), mparams
        )
        if not self._model:
            self._teardown()
            raise LibmtmdError(f"failed to load model: {self._model_path}")
        self._vocab = llama.llama_model_get_vocab(self._model)

        # Diagnostic: a model with NO embedded chat template is a base
        # (pretrained) model, not an instruction-tuned one. Base models do not
        # understand the Gemma 4 "<|turn>" chat format and emit unrelated
        # continuation (e.g. markdown/html) instead of an answer.
        tmpl = llama.llama_model_chat_template(self._model, None)
        if not tmpl:
            logger.warning(
                "model has NO embedded chat template -- this looks like a base "
                "(non-instruction-tuned) model, not an instruct model; the chat "
                "format below will not be understood"
            )
        else:
            logger.info("model chat template present (len=%d)", len(tmpl))

        # -- context ----------------------------------------------------------
        cparams = llama.llama_context_default_params()
        cparams.n_ctx = self._n_ctx
        cparams.n_batch = self._n_batch
        cparams.n_ubatch = self._n_ubatch
        if self._n_threads is not None:
            cparams.n_threads = self._n_threads
            cparams.n_threads_batch = self._n_threads
        self._lctx = llama.llama_init_from_model(self._model, cparams)
        if not self._lctx:
            self._teardown()
            raise LibmtmdError("failed to create llama context")
        self._n_batch = int(llama.llama_n_batch(self._lctx))

        # -- multimodal projector (mmproj) -----------------------------------
        mctx_params = mtmd.mtmd_context_params_default()
        mctx_params.use_gpu = self._mmproj_use_gpu
        mctx_params.print_timings = self._verbose
        if self._n_threads is not None:
            mctx_params.n_threads = self._n_threads
        self._mtmd_ctx = mtmd.mtmd_init_from_file(
            os.fsencode(self._mmproj_path), self._model, mctx_params
        )
        if not self._mtmd_ctx:
            self._teardown()
            raise LibmtmdError(f"failed to load mmproj: {self._mmproj_path}")

        self._media_marker = mtmd.mtmd_default_marker()

        # -- sampler chain (mirrors common/sampling.cpp default order:
        #    penalties -> ... -> top_p -> ... -> temperature -> dist) --------
        smpl = llama.llama_sampler_chain_init(llama.llama_sampler_chain_default_params())
        n_vocab = llama.llama_vocab_n_tokens(self._vocab)
        llama.llama_sampler_chain_add(
            smpl,
            llama.llama_sampler_init_penalties(
                n_vocab, 64, self._repeat_penalty, 0.0, 0.0
            ),
        )
        llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_top_p(self._top_p, 1))
        llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_temp(self._temperature))
        llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_dist(self._seed))
        self._smpl = smpl
        logger.info(
            "sampler chain initialized: repeat_penalty=%.2f, top_p=%.2f, temperature=%.2f",
            self._repeat_penalty, self._top_p, self._temperature,
        )

        self._loaded = True
        logger.info(
            "NativeMTMDBackend loaded (model=%s, mmproj=%s, n_ctx=%d, n_gpu_layers=%d)",
            self._model_path, self._mmproj_path, self._n_ctx, self._n_gpu_layers,
        )

    def _teardown(self) -> None:
        llama = self._llama
        mtmd = self._mtmd
        if llama is not None and self._smpl is not None:
            llama.llama_sampler_free(self._smpl)
            self._smpl = None
        if mtmd is not None and self._mtmd_ctx is not None:
            mtmd.mtmd_free(self._mtmd_ctx)
            self._mtmd_ctx = None
        if llama is not None and self._lctx is not None:
            llama.llama_free(self._lctx)
            self._lctx = None
        if llama is not None and self._model is not None:
            llama.llama_model_free(self._model)
            self._model = None
        self._loaded = False

    def close(self) -> None:
        """Release native resources. Safe to call multiple times."""
        self._teardown()

    def __enter__(self) -> "NativeMTMDBackend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start_session(self) -> None:
        """Begin a fresh session: clear the KV cache, sampler and counters.

        This is the only operation (alongside :meth:`end_session`) that clears
        session-level runtime state. It must NOT be called between chunks of the
        same session. No-op-safe before the first load.
        """
        self._clear_session_state()
        self._session_active = True

    def end_session(self) -> None:
        """Terminate the whole video interaction and clear session state.

        Equivalent to "start a brand-new session". Unlike :meth:`stop_query`,
        this clears the native KV memory and resets the sampler.
        """
        self._clear_session_state()
        self._session_active = False

    def reset(self) -> None:
        """Backwards-compatible alias for :meth:`end_session`."""
        self.end_session()

    def set_metrics_recorder(self, recorder) -> None:
        """Attach a benchmark metrics sink (duck-typed: ``record_append(...)``).

        The benchmark driver calls this once per video so that
        :meth:`_append_to_context` can report per-append KV/context metrics and
        per-frame visual-token counts. It is pure observation and never changes
        inference behaviour or the KV-cache lifecycle.
        """
        self._metrics_recorder = recorder

    def _install_log_filters(self) -> None:
        """Redirect libmtmd/libllama verbose logging to the warn/error filter.
        """
        mtmd_set = getattr(self._mtmd, "mtmd_log_set", None)
        if mtmd_set is not None:
            mtmd_set(_LOG_FILTER, None)
        helper_set = getattr(self._mtmd, "mtmd_helper_log_set", None)
        if helper_set is not None:
            helper_set(_LOG_FILTER, None)
        llama_set = getattr(self._llama, "llama_log_set", None)
        if llama_set is not None:
            llama_set(_LOG_FILTER, None)

    def _clear_session_state(self) -> None:
        self._n_past = 0
        self._session_active = False
        self._query_active = False
        self._turn_active = False
        self._model_turn_open = False
        self._query_pieces = []
        self._query_tokens = 0
        self._n_frames_added = 0
        self._generation_count = 0
        self._logits_logged = False
        self._last_context_token_id = None
        if self._loaded and self._llama is not None and self._lctx is not None:
            self._llama.llama_memory_clear(
                self._llama.llama_get_memory(self._lctx), True
            )
            if self._smpl is not None:
                self._llama.llama_sampler_reset(self._smpl)

    # -- inference -----------------------------------------------------------

    def infer(self, rgb: bytes, width: int, height: int, prompt: str) -> str:
        """One-shot single-image QA through the streaming session.

        This is the single-image path (``scripts/run_image.py``): start a fresh
        session, append the image as one visual chunk, then run a single query
        to completion. The session is left active for the caller to end.
        """
        self.start_session()
        self.append_video_chunk([(rgb, width, height)])
        return self.finalize(prompt)

    def append_video_chunk(self, frames: list[tuple[bytes, int, int]]) -> None:
        """Append a *chunk* of video frames to the persistent context.

        ``frames`` is a list of ``(rgb_bytes, width, height)`` tuples processed
        as ONE ``mtmd_tokenize`` + ONE ``mtmd_helper_eval_chunks`` call,
        advancing ``self._n_past`` without generating.

        This is allowed both before and DURING an active query:

        * Before a query (or between queries), the first chunk opens the Gemma 4
          user turn (``<|turn>user\\n`` + BOS) and later chunks continue it, so
          frames accumulate as the user turn.
        * During an active query, the media is injected directly at the current
          context position (no new turn) and the eval puts ``logits_last`` on the
          final media token, so the in-flight generation continues from the
          updated KV. The context is never reset.
        """
        self._ensure_loaded()
        if not self._session_active:
            raise RuntimeError(
                "append_video_chunk() requires an active session; "
                "call start_session() first"
            )
        assert self._mtmd is not None and self._llama is not None
        if not frames:
            return

        marker = self._media_marker.decode("utf-8")

        if self._query_active:
            # Mid-generation: append media at the current context position and
            # leave the model turn open. logits_last=True so step_generation()
            # continues from the media's final token.
            text = marker + ("\n" + marker) * (len(frames) - 1)
            n_pos = self._append_to_context(
                text, list(frames), add_special=False, logits_last=True,
            )
        else:
            self._close_model_turn_if_open()
            first = not self._turn_active
            prefix = _GEMMA4_USER_OPEN if first else "\n"
            text = prefix + marker + ("\n" + marker) * (len(frames) - 1)
            n_pos = self._append_to_context(
                text, list(frames), add_special=(self._n_past == 0), logits_last=False,
            )
            self._turn_active = True

        self._n_frames_added += len(frames)
        logger.debug(
            "[session] visual chunk appended: frames=%d, n_pos=%d, n_past=%d",
            len(frames), n_pos, self._n_past,
        )

    def add_frame(self, rgb: bytes, width: int, height: int) -> None:
        """Append one video frame (thin wrapper over :meth:`append_video_chunk`)."""
        self.append_video_chunk([(rgb, width, height)])

    def start_query(self, question: str) -> None:
        """Open a query: close the user turn, open the model turn, no generation.

        Closes the current user turn with ``question<turn|>``, opens
        ``<|turn>model``, and evals with ``logits_last=True`` so
        :meth:`step_generation` can sample. Never resets the session; the
        accumulated visual/KV state is preserved so a later query can reuse it.
        """
        if self._query_active:
            raise RuntimeError(
                "start_query() called while a query is already active"
            )
        self._ensure_loaded()
        if not self._session_active:
            raise RuntimeError(
                "start_query() requires an active session; call start_session() first"
            )
        assert self._mtmd is not None and self._llama is not None

        self._close_model_turn_if_open()

        prefix = "" if self._turn_active else _GEMMA4_USER_OPEN
        text = (
            prefix + question + _GEMMA4_USER_CLOSE
            + _GEMMA4_MODEL_OPEN + _GEMMA4_THINK_SUFFIX
        )
        self._append_to_context(
            text, [], add_special=(self._n_past == 0), logits_last=True
        )

        self._turn_active = False
        self._model_turn_open = True
        self._query_active = True
        self._query_pieces = []
        self._query_tokens = 0
        self._generation_count += 1
        logger.debug("[query] started: n_past=%d", self._n_past)

    def stop_query(self) -> None:
        """Stop the current query without touching session/KV state.

        Terminates the query at this (serial) scheduling boundary and returns it
        to idle. The KV cache, model context, accumulated visual history and
        sampler are all preserved. If generation was cut off mid-turn, the model
        turn is left open; the next turn-boundary operation closes it with the
        official ``<turn|>`` marker (see ``_close_model_turn_if_open``).
        """
        if not self._query_active:
            return
        self._query_active = False
        logger.debug("[query] stopped at n_past=%d (session preserved)", self._n_past)

    def step_generation(self) -> tuple[str, bool]:
        """Advance the active query by exactly one token.

        Returns ``(text, done)`` where ``text`` is the query's full generated
        text so far (markers stripped) and ``done`` is ``True`` when generation
        finished: end-of-turn / end-of-generation reached, or the per-query
        ``max_tokens`` cap hit. The model turn is closed on a natural end
        (EOG/EOT); when the ``max_tokens`` cap truncates generation the model
        turn is left open (like :meth:`stop_query`) so the next turn-boundary
        operation closes it. The cap never resets the session/KV.
        """
        assert self._llama is not None
        if not self._query_active:
            raise RuntimeError("step_generation() called with no active query")

        if not self._logits_logged:
            if self._verbose:
                self._logits_diagnostic()
            self._logits_logged = True

        token = self._sample_one_token()
        self._query_tokens += 1
        eot_token = self._llama.llama_vocab_eot(self._vocab)

        if self._llama.llama_vocab_is_eog(self._vocab, token) or token == eot_token:
            self._query_active = False
            self._model_turn_open = False
            return "".join(self._query_pieces).strip(), True

        piece = self._token_to_piece(token)
        self._query_pieces.append(piece)
        text = "".join(self._query_pieces)

        # Defensive fallback: when the vocab/lib does not mark the end-of-turn
        # marker EOG/control it reaches us as literal text, split across several
        # tokens. Match against the accumulated text and stop once complete.
        for marker in _EOT_TEXT_MARKERS:
            pos = text.rfind(marker, max(0, len(text) - 64))
            if pos >= 0:
                self._query_active = False
                self._model_turn_open = False
                return text[:pos].strip(), True

        # Per-query ``max_tokens`` cap: end the query once it has emitted this
        # many tokens without an EOG/EOT. This bounds the streaming path (e.g.
        # ``Agent.run()`` looping on ``step_generation()``) when the model never
        # emits an end-of-turn token. The cap is counted per query (reset by
        # ``start_query()``), NOT per session. The model turn is left open so
        # the truncated turn is closed by the next turn-boundary operation, as
        # for ``stop_query()``; the KV/session is never reset.
        if self._query_tokens >= self._max_tokens:
            self._query_active = False
            return text.strip(), True

        return text.strip(), False

    def finalize(self, question: str) -> str:
        """Run a single query to completion (convenience wrapper).

        This is no longer the architectural center; it is a thin wrapper over
        :meth:`start_query` + a :meth:`step_generation` loop, kept for the
        single-image / single-question callers.
        """
        self.start_query(question)
        return self._generate()

    def _sample_one_token(self) -> int:
        """Sample one token and decode it into the persistent KV cache.

        Sampler state note (kept explicit because a version bump here has broken
        this before): in the llama.cpp version vendored at
        ``third_party/llama.cpp-master``, ``llama_sampler_sample()`` already
        *accepts* the sampled token into the sampler chain internally — it is
        shorthand for "apply + pick + ``llama_sampler_accept``". See
        ``src/llama-sampler.cpp`` (the trailing ``llama_sampler_accept(smpl,
        token)`` before ``return token``) and the ``include/llama.h`` docstring
        for ``llama_sampler_sample``. We therefore must NOT call
        ``llama_sampler_accept()`` here: a second accept would double-update the
        repetition-penalty state. If a future llama.cpp version changes this
        contract, the accept must be added here instead.
        """
        assert self._llama is not None
        llama = self._llama
        token = llama.llama_sampler_sample(self._smpl, self._lctx, -1)

        token_arr = (ctypes.c_int32 * 1)(token)
        pos_arr = (ctypes.c_int32 * 1)(self._n_past)
        n_seq_id = (ctypes.c_int32 * 1)(1)
        seq_id = (ctypes.c_int32 * 1)(0)
        seq_id_ptr = (ctypes.POINTER(ctypes.c_int32) * 1)(seq_id)
        logits = (ctypes.c_int8 * 1)(1)
        batch = LlamaBatch(
            n_tokens=1,
            token=token_arr,
            embd=None,
            pos=pos_arr,
            n_seq_id=n_seq_id,
            seq_id=seq_id_ptr,
            logits=logits,
        )
        res = llama.llama_decode(self._lctx, batch)
        if res != 0:
            raise LibmtmdError(f"llama_decode failed: {res}")
        self._n_past += 1
        return token

    def _close_model_turn_if_open(self) -> None:
        """Close a model turn left open by stop_query().

        If a query was stopped mid-generation, the ``<|turn>model`` turn has no
        terminating ``<turn|>`` yet. Gemma 4 requires every message to end with
        ``<turn|>``, so before opening the next user turn we append the official
        end-of-turn marker (exactly what the reference re-renders when it
        reformats history). No generation happens here.
        """
        if not self._model_turn_open:
            return
        self._append_to_context(
            _GEMMA4_TURN_CLOSE, [], add_special=False, logits_last=False
        )
        self._model_turn_open = False

    def infer_cli_equivalent(self, rgb: bytes, width: int, height: int, question: str) -> str:
        """Single-image, CLI-equivalent baseline (debug path).

        Mirrors llama-mtmd-cli's single-turn flow exactly: one
        ``mtmd_tokenize_from_parts`` over ``[TEXT, IMAGE, TEXT]``, one
        ``mtmd_helper_eval_chunks``, one generation. The image is never eval'd
        separately and never enters the KV cache before the whole multimodal
        context is tokenized, so this isolates whether Native's two-call
        ``mtmd_tokenize`` path differs from the CLI's one-call parts path.

        Text parts use the same Gemma 4 turn structure as ``finalize()``: the
        CLI's enable_thinking=true rendering (leading ``<|think|>`` system turn,
        thought channel left open). The ``<|image>`` / ``<image|>`` delimiters
        are added automatically by the projector around the image chunk, exactly
        as in the CLI.
        """
        self.start_session()
        self._ensure_loaded()
        assert self._mtmd is not None and self._llama is not None
        mtmd = self._mtmd

        text_before = _GEMMA4_USER_OPEN.encode("utf-8")  # "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n"
        text_after = (
            question + _GEMMA4_USER_CLOSE + _GEMMA4_MODEL_OPEN + _GEMMA4_THINK_SUFFIX
        ).encode("utf-8")

        data = (ctypes.c_uint8 * len(rgb)).from_buffer_copy(rgb)
        bitmap = mtmd.mtmd_bitmap_init(width, height, data)
        if not bitmap:
            raise LibmtmdError("mtmd_bitmap_init failed")

        chunks = mtmd.mtmd_input_chunks_init()
        if not chunks:
            mtmd.mtmd_bitmap_free(bitmap)
            raise LibmtmdError("mtmd_input_chunks_init failed")
        try:
            # Three parts, in the exact order the CLI's eval_message() builds
            # them: TEXT(before marker), IMAGE, TEXT(after marker). Per-part
            # add_special is ignored by mtmd_tokenize_from_parts(); the single
            # BOS is requested by the trailing add_special=true (add_bos).
            text0 = MtmdInputText(
                text=text_before, text_len=len(text_before),
                add_special=False, parse_special=True,
            )
            text1 = MtmdInputText(
                text=text_after, text_len=len(text_after),
                add_special=False, parse_special=True,
            )
            part0 = MtmdInputPart(text=ctypes.pointer(text0), bitmap=None)
            part1 = MtmdInputPart(text=None, bitmap=bitmap)
            part2 = MtmdInputPart(text=ctypes.pointer(text1), bitmap=None)
            parts = (ctypes.POINTER(MtmdInputPart) * 3)(
                ctypes.pointer(part0), ctypes.pointer(part1), ctypes.pointer(part2),
            )

            res = mtmd.mtmd_tokenize_from_parts(self._mtmd_ctx, chunks, parts, 3, True)
            if res != 0:
                raise LibmtmdError(f"mtmd_tokenize_from_parts failed: {res}")

            n_pos = mtmd.mtmd_helper_get_n_pos(chunks)
            if self._n_past + n_pos > self._n_ctx:
                raise LibmtmdError(
                    f"image context exceeds model context window: "
                    f"n_past={self._n_past} + n_pos={n_pos} > n_ctx={self._n_ctx}."
                )

            if self._verbose:
                self._record_last_text_token(chunks)
                self._dump_chunks(chunks)

            self._n_past = self._eval_chunks(chunks, logits_last=True)
            logger.info("[cli-equivalent] single-image context eval'd; n_past=%d", self._n_past)
            self._turn_active = False
            self._model_turn_open = True
            self._query_active = True
            self._query_pieces = []
            self._generation_count += 1
            output = self._generate()
            logger.info(
                "[cli-equivalent] single-image generation finished; answer length=%d",
                len(output),
            )
            return output
        finally:
            mtmd.mtmd_input_chunks_free(chunks)
            mtmd.mtmd_bitmap_free(bitmap)

    def _collect_image_token_counts(self, chunks: int) -> list[int]:
        """Per-image visual-token counts from the tokenized chunks, in order.

        ``MTMD_INPUT_CHUNK_TYPE_IMAGE == 1`` (see mtmd.h). This is the *actual*
        number of visual tokens the vision encoder produced for each frame, read
        straight from the tokenizer output -- never derived from resolution.
        """
        mtmd = self._mtmd
        counts: list[int] = []
        n = mtmd.mtmd_input_chunks_size(chunks)
        for i in range(n):
            chunk = mtmd.mtmd_input_chunks_get(chunks, i)
            if mtmd.mtmd_input_chunk_get_type(chunk) == 1:
                counts.append(int(mtmd.mtmd_input_chunk_get_n_tokens(chunk)))
        return counts

    def _append_to_context(
        self,
        text: str,
        frames: list[tuple[bytes, int, int]],
        add_special: bool,
        logits_last: bool,
    ) -> int:
        """Tokenize ``text`` (+ media ``frames``) and evaluate it into the cache.

        Returns the number of positions the new chunks occupy (``n_pos``), used
        for logging and the context-overflow check.

        Context overflow is deterministic: when ``n_past + n_pos`` would exceed
        ``n_ctx`` we raise :class:`LibmtmdError` *before* evaluating the chunks,
        so ``n_past`` and the KV cache are left exactly as they were. We never
        silently truncate the KV or reset the session to make room.

        TODO(eviction): a long-lived streaming session accumulates visual/text
        KV without bound, so a real deployment needs an eviction/retention
        policy (or a sliding context window). That is intentionally not
        implemented here; until it is, reaching ``n_ctx`` is a hard error.
        """
        assert self._mtmd is not None and self._llama is not None
        mtmd = self._mtmd
        n_frames = len(frames)

        bitmaps: list[int] = []
        try:
            for rgb, width, height in frames:
                data = (ctypes.c_uint8 * len(rgb)).from_buffer_copy(rgb)
                bitmap = mtmd.mtmd_bitmap_init(width, height, data)
                if not bitmap:
                    raise LibmtmdError("mtmd_bitmap_init failed")
                bitmaps.append(bitmap)

            chunks = mtmd.mtmd_input_chunks_init()
            if not chunks:
                raise LibmtmdError("mtmd_input_chunks_init failed")
            try:
                text_bytes = text.encode("utf-8")
                input_text = MtmdInputText(
                    text=text_bytes,
                    text_len=len(text_bytes),
                    add_special=add_special,
                    parse_special=True,
                )
                bitmap_arr = (
                    (ctypes.c_void_p * n_frames)(*bitmaps) if n_frames else None
                )
                res = mtmd.mtmd_tokenize(
                    self._mtmd_ctx, chunks, ctypes.byref(input_text), bitmap_arr, n_frames
                )
                if res != 0:
                    raise LibmtmdError(f"mtmd_tokenize failed: {res}")

                n_pos = mtmd.mtmd_helper_get_n_pos(chunks)
                image_token_counts = self._collect_image_token_counts(chunks)

                # Diagnostic only (verbose): remember the last text token ID
                # (what the model is conditioned on) and dump the tokenized
                # chunks, so the server can diff Native's token sequence against
                # llama-mtmd-cli. Both are no-ops unless verbose is enabled.
                if self._verbose:
                    self._record_last_text_token(chunks)
                    self._dump_chunks(chunks)

                # Deterministic overflow boundary: raise before eval, so n_past
                # and the KV are unchanged. No eviction/retention policy exists
                # yet (see the docstring TODO); never silently truncate/reset.
                if self._n_past + n_pos > self._n_ctx:
                    raise LibmtmdError(
                        f"video context exceeds model context window: "
                        f"n_past={self._n_past} + n_pos={n_pos} > n_ctx={self._n_ctx}. "
                        f"Call reset() to start a new video."
                    )

                self._n_past = self._eval_chunks(chunks, logits_last=logits_last)

                if self._metrics_recorder is not None:
                    images = [
                        (frames[i][1], frames[i][2], image_token_counts[i])
                        for i in range(min(len(frames), len(image_token_counts)))
                    ]
                    self._metrics_recorder.record_append(
                        wall_time=time.time(),
                        n_pos_added=n_pos,
                        kv_cache_length_tokens=self._n_past,
                        images=images,
                    )

                return n_pos
            finally:
                mtmd.mtmd_input_chunks_free(chunks)
        finally:
            for bitmap in bitmaps:
                mtmd.mtmd_bitmap_free(bitmap)

    def _eval_chunks(self, chunks: int, logits_last: bool) -> int:
        """Decode all tokenized chunks; returns the resulting ``n_past``.

        Starts from ``self._n_past`` so this frame's tokens are appended to the
        existing context (not overwriting it). ``mtmd_helper_eval_chunks``
        (libmtmd) handles the per-chunk dispatch: text chunks are decoded
        directly, image/audio chunks are encoded and decoded internally.
        ``logits_last`` puts the output logits on the final chunk so the sampler
        can read them; only the last eval before generation needs it.
        """
        assert self._mtmd is not None and self._llama is not None

        new_n_past = ctypes.c_int32(0)
        res = self._mtmd.mtmd_helper_eval_chunks(
            self._mtmd_ctx, self._lctx, chunks, self._n_past, 0,
            self._n_batch, logits_last, ctypes.pointer(new_n_past),
        )
        if res != 0:
            raise LibmtmdError(f"mtmd_helper_eval_chunks failed: {res}")
        return new_n_past.value

    def _logits_diagnostic(self) -> None:
        """Log the pre-sample logits once, to separate a prefill/logits error
        from a sampler error.
        """
        assert self._llama is not None
        llama = self._llama

        try:
            n_vocab = llama.llama_vocab_n_tokens(self._vocab)
            logits_ptr = llama.llama_get_logits_ith(self._lctx, -1)
            if not logits_ptr:
                logger.warning(
                    "logits diagnostic: llama_get_logits_ith(ctx, -1) returned NULL"
                )
                return

            logits = logits_ptr[:n_vocab]
            top20 = sorted(
                ((logit, tid) for tid, logit in enumerate(logits)), reverse=True
            )[:20]

            last_tid = self._last_context_token_id
            last_piece = self._token_to_piece(last_tid) if last_tid is not None else None
            logger.info(
                "logits diagnostic: n_past=%d n_vocab=%d last_token_id=%s last_token_piece=%r",
                self._n_past, n_vocab, last_tid, last_piece,
            )
            for rank, (logit, tid) in enumerate(top20, start=1):
                logger.info(
                    "  top-20[%02d] token_id=%d piece=%r logit=%.4f",
                    rank, tid, self._token_to_piece(tid), logit,
                )
        except Exception as exc:  # diagnostic must never break generation
            logger.warning("logits diagnostic skipped: %s", exc)

    def _record_last_text_token(self, chunks: int) -> None:
        """Diagnostic only: remember the last text token ID in ``chunks``.

        This is the token the model is conditioned on right before sampling, so
        ``_logits_diagnostic()`` can label the pre-sample logits. Never fails.
        """
        assert self._mtmd is not None
        mtmd = self._mtmd
        self._last_context_token_id = None
        try:
            n_chunks = int(mtmd.mtmd_input_chunks_size(chunks))
            for i in range(n_chunks - 1, -1, -1):
                chunk = mtmd.mtmd_input_chunks_get(chunks, i)
                n_tokens = ctypes.c_size_t(0)
                tokens = mtmd.mtmd_input_chunk_get_tokens_text(
                    chunk, ctypes.byref(n_tokens)
                )
                if tokens and n_tokens.value > 0:
                    self._last_context_token_id = int(tokens[n_tokens.value - 1])
                    break
        except Exception as exc:  # diagnostic must never break generation
            logger.warning("last-text-token scan skipped: %s", exc)

    def _dump_chunks(self, chunks: int) -> None:
        """Diagnostic only: dump the tokenized chunk order and text token IDs.
        """
        assert self._mtmd is not None
        mtmd = self._mtmd
        try:
            n_chunks = int(mtmd.mtmd_input_chunks_size(chunks))
            logger.info(
                "token dump: n_chunks=%d (n_past_before=%d)", n_chunks, self._n_past
            )
            for i in range(n_chunks):
                chunk = mtmd.mtmd_input_chunks_get(chunks, i)
                ctype = int(mtmd.mtmd_input_chunk_get_type(chunk))
                if ctype == 0:  # MTMD_INPUT_CHUNK_TYPE_TEXT
                    n_tokens = ctypes.c_size_t(0)
                    tokens = mtmd.mtmd_input_chunk_get_tokens_text(
                        chunk, ctypes.byref(n_tokens)
                    )
                    entries = [
                        f"{int(tokens[j])}:{self._token_to_piece(int(tokens[j]))!r}"
                        for j in range(n_tokens.value)
                    ]
                    logger.info(
                        "  chunk[%d] TEXT n_tokens=%d %s",
                        i, n_tokens.value, " ".join(entries),
                    )
                else:
                    kind = {1: "IMAGE", 2: "AUDIO"}.get(ctype, f"UNKNOWN({ctype})")
                    logger.info(
                        "  chunk[%d] %s n_tokens=%d",
                        i, kind, int(mtmd.mtmd_input_chunk_get_n_tokens(chunk)),
                    )
        except Exception as exc:  # diagnostic must never break generation
            logger.warning("token dump skipped: %s", exc)

    def _generate(self) -> str:
        """Run the active query to completion (loop :meth:`step_generation`)."""
        if not self._query_active:
            raise RuntimeError("_generate() called with no active query")
        text = ""
        for _ in range(self._max_tokens):
            text, done = self.step_generation()
            if done:
                break
        return text

    def _token_to_piece(self, token: int) -> str:
        assert self._llama is not None
        size = 64
        while True:
            buf = ctypes.create_string_buffer(size)
            n = self._llama.llama_token_to_piece(
                self._vocab, token, buf, size, 0, False
            )
            if n < 0:
                # Buffer too small: -n is the required length.
                size = -n + 1
                continue
            return buf.raw[:n].decode("utf-8", errors="replace")
