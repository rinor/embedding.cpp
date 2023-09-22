#ifndef BERT_GGUF_H
#define BERT_GGUF_H

#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <thread>
#include <algorithm>

enum bert_token_type
{
    LLAMA_TOKEN_TYPE_UNDEFINED = 0,
    LLAMA_TOKEN_TYPE_NORMAL = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN = 2,
    LLAMA_TOKEN_TYPE_CONTROL = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED = 5,
    LLAMA_TOKEN_TYPE_BYTE = 6,
};

//
// gguf helpers
//

// LLAMA_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char *fmt, ...)
{
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

#define GGUF_GET_KEY(ctx, dst, func, type, req, key)                                                                \
    {                                                                                                               \
        const std::string skey(key);                                                                                \
        const int kid = gguf_find_key(ctx, skey.c_str());                                                           \
        if (kid >= 0)                                                                                               \
        {                                                                                                           \
            enum gguf_type ktype = gguf_get_kv_type(ctx, kid);                                                      \
            if (ktype != (type))                                                                                    \
            {                                                                                                       \
                throw std::runtime_error(format("key %s has wrong type: %s", skey.c_str(), gguf_type_name(ktype))); \
            }                                                                                                       \
            (dst) = func(ctx, kid);                                                                                 \
        }                                                                                                           \
        else if (req)                                                                                               \
        {                                                                                                           \
            throw std::runtime_error(format("key not found in model: %s", skey.c_str()));                           \
        }                                                                                                           \
    }

//
// gguf constants (sync with gguf.py)
//

enum llm_arch
{
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_BERT,
    LLM_ARCH_UNKNOWN,
};

static std::map<llm_arch, std::string> LLM_ARCH_NAMES = {
    {LLM_ARCH_LLAMA, "llama"},
    {LLM_ARCH_FALCON, "falcon"},
    {LLM_ARCH_GPT2, "gpt2"},
    {LLM_ARCH_GPTJ, "gptj"},
    {LLM_ARCH_GPTNEOX, "gptneox"},
    {LLM_ARCH_MPT, "mpt"},
    {LLM_ARCH_BAICHUAN, "baichuan"},
    {LLM_ARCH_STARCODER, "starcoder"},
    {LLM_ARCH_BERT, "bert"},
};

enum llm_kv
{
    LLM_KV_GENERAL_ARCHITECTURE,
    LLM_KV_GENERAL_QUANTIZATION_VERSION,
    LLM_KV_GENERAL_ALIGNMENT,
    LLM_KV_GENERAL_NAME,
    LLM_KV_GENERAL_AUTHOR,
    LLM_KV_GENERAL_URL,
    LLM_KV_GENERAL_DESCRIPTION,
    LLM_KV_GENERAL_LICENSE,
    LLM_KV_GENERAL_SOURCE_URL,
    LLM_KV_GENERAL_SOURCE_HF_REPO,

    LLM_KV_CONTEXT_LENGTH,
    LLM_KV_EMBEDDING_LENGTH,
    LLM_KV_BLOCK_COUNT,
    LLM_KV_FEED_FORWARD_LENGTH,
    LLM_KV_USE_PARALLEL_RESIDUAL,
    LLM_KV_TENSOR_DATA_LAYOUT,

    LLM_KV_ATTENTION_HEAD_COUNT,
    LLM_KV_ATTENTION_HEAD_COUNT_KV,
    LLM_KV_ATTENTION_MAX_ALIBI_BIAS,
    LLM_KV_ATTENTION_CLAMP_KQV,
    LLM_KV_ATTENTION_LAYERNORM_EPS,
    LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,

    LLM_KV_ROPE_DIMENSION_COUNT,
    LLM_KV_ROPE_FREQ_BASE,
    LLM_KV_ROPE_SCALE_LINEAR,

    LLM_KV_TOKENIZER_MODEL,
    LLM_KV_TOKENIZER_LIST,
    LLM_KV_TOKENIZER_TOKEN_TYPE,
    LLM_KV_TOKENIZER_SCORES,
    LLM_KV_TOKENIZER_MERGES,
    LLM_KV_TOKENIZER_BOS_ID,
    LLM_KV_TOKENIZER_EOS_ID,
    LLM_KV_TOKENIZER_UNK_ID,
    LLM_KV_TOKENIZER_SEP_ID,
    LLM_KV_TOKENIZER_PAD_ID,
    LLM_KV_TOKENIZER_HF_JSON,
    LLM_KV_TOKENIZER_RWKV,
};

static std::map<llm_kv, std::string> LLM_KV_NAMES = {
    {LLM_KV_GENERAL_ARCHITECTURE, "general.architecture"},
    {LLM_KV_GENERAL_QUANTIZATION_VERSION, "general.quantization_version"},
    {LLM_KV_GENERAL_ALIGNMENT, "general.alignment"},
    {LLM_KV_GENERAL_NAME, "general.name"},
    {LLM_KV_GENERAL_AUTHOR, "general.author"},
    {LLM_KV_GENERAL_URL, "general.url"},
    {LLM_KV_GENERAL_DESCRIPTION, "general.description"},
    {LLM_KV_GENERAL_LICENSE, "general.license"},
    {LLM_KV_GENERAL_SOURCE_URL, "general.source_url"},
    {LLM_KV_GENERAL_SOURCE_HF_REPO, "general.source_hf_repo"},

    {LLM_KV_CONTEXT_LENGTH, "%s.context_length"},
    {LLM_KV_EMBEDDING_LENGTH, "%s.embedding_length"},
    {LLM_KV_BLOCK_COUNT, "%s.block_count"},
    {LLM_KV_FEED_FORWARD_LENGTH, "%s.feed_forward_length"},
    {LLM_KV_USE_PARALLEL_RESIDUAL, "%s.use_parallel_residual"},
    {LLM_KV_TENSOR_DATA_LAYOUT, "%s.tensor_data_layout"},

    {LLM_KV_ATTENTION_HEAD_COUNT, "%s.attention.head_count"},
    {LLM_KV_ATTENTION_HEAD_COUNT_KV, "%s.attention.head_count_kv"},
    {LLM_KV_ATTENTION_MAX_ALIBI_BIAS, "%s.attention.max_alibi_bias"},
    {LLM_KV_ATTENTION_CLAMP_KQV, "%s.attention.clamp_kqv"},
    {LLM_KV_ATTENTION_LAYERNORM_EPS, "%s.attention.layer_norm_epsilon"},
    {LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, "%s.attention.layer_norm_rms_epsilon"},

    {LLM_KV_ROPE_DIMENSION_COUNT, "%s.rope.dimension_count"},
    {LLM_KV_ROPE_FREQ_BASE, "%s.rope.freq_base"},
    {LLM_KV_ROPE_SCALE_LINEAR, "%s.rope.scale_linear"},

    {LLM_KV_TOKENIZER_MODEL, "tokenizer.ggml.model"},
    {LLM_KV_TOKENIZER_LIST, "tokenizer.ggml.tokens"},
    {LLM_KV_TOKENIZER_TOKEN_TYPE, "tokenizer.ggml.token_type"},
    {LLM_KV_TOKENIZER_SCORES, "tokenizer.ggml.scores"},
    {LLM_KV_TOKENIZER_MERGES, "tokenizer.ggml.merges"},
    {LLM_KV_TOKENIZER_BOS_ID, "tokenizer.ggml.bos_token_id"},
    {LLM_KV_TOKENIZER_EOS_ID, "tokenizer.ggml.eos_token_id"},
    {LLM_KV_TOKENIZER_UNK_ID, "tokenizer.ggml.unknown_token_id"},
    {LLM_KV_TOKENIZER_SEP_ID, "tokenizer.ggml.seperator_token_id"},
    {LLM_KV_TOKENIZER_PAD_ID, "tokenizer.ggml.padding_token_id"},
    {LLM_KV_TOKENIZER_HF_JSON, "tokenizer.huggingface.json"},
    {LLM_KV_TOKENIZER_RWKV, "tokenizer.rwkv.world"},
};

struct LLM_KV
{
    LLM_KV(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    std::string operator()(llm_kv kv) const
    {
        return ::format(LLM_KV_NAMES[kv].c_str(), LLM_ARCH_NAMES[arch].c_str());
    }
};

#endif // BERT_GGUF_H