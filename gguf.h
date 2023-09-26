#ifndef GGUF_H
#define GGUF_H

#include "ggml.h"

#include <cassert>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#ifdef __GNUC__
#ifdef __MINGW32__
#define GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define GGML_ATTRIBUTE_FORMAT(...)
#endif

GGML_ATTRIBUTE_FORMAT(1, 2)
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

// a copy from `llama.cpp` named `llama_file`
struct gguf_file
{
    // use FILE * so we don't have to re-open the file to mmap
    FILE *fp;
    size_t size;

    gguf_file(const char *fname, const char *mode)
    {
        fp = std::fopen(fname, mode);
        if (fp == NULL)
        {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const
    {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t)ret;
    }

    void seek(size_t offset, int whence) const
    {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64)offset, whence);
#else
        int ret = std::fseek(fp, (long)offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    void read_raw(void *ptr, size_t len) const
    {
        if (len == 0)
        {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp))
        {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1)
        {
            throw std::runtime_error(std::string("unexpectedly reached end of file"));
        }
    }

    uint32_t read_u32() const
    {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void *ptr, size_t len) const
    {
        if (len == 0)
        {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1)
        {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) const
    {
        write_raw(&val, sizeof(val));
    }

    ~gguf_file()
    {
        if (fp)
        {
            std::fclose(fp);
        }
    }
};

// sync from gguf python
enum gguf_token_type
{
    GGUF_TOKEN_TYPE_UNDEFINED = 0,
    GGUF_TOKEN_TYPE_NORMAL = 1,
    GGUF_TOKEN_TYPE_UNKNOWN = 2,
    GGUF_TOKEN_TYPE_CONTROL = 3,
    GGUF_TOKEN_TYPE_USER_DEFINED = 4,
    GGUF_TOKEN_TYPE_UNUSED = 5,
    GGUF_TOKEN_TYPE_BYTE = 6,
};

//
// gguf helpers
//

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
    LLM_ARCH_GGML,
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
    {LLM_ARCH_GGML, "llama"},
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

    // FIXME: add for embedding only
    LLM_KV_TOKENIZER_CLS_ID
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

    // FIXME: add for embedding only
    {LLM_KV_TOKENIZER_CLS_ID, "tokenizer.ggml.cls_token_id"},

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

enum gguf_fver
{
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
};

static const char *llama_file_version_name(gguf_fver version)
{
    switch (version)
    {
    case GGUF_FILE_VERSION_V1:
        return "GGUF V1 (support until nov 2023)";
    case GGUF_FILE_VERSION_V2:
        return "GGUF V2 (latest)";
    }

    return "unknown";
}

static std::string llama_format_tensor_shape(const std::vector<int64_t> &ne)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++)
    {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, ne.at(i));
    }
    return buf;
}

static std::string llama_format_tensor_shape(const struct ggml_tensor *t)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++)
    {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, t->ne[i]);
    }
    return buf;
}

#endif // GGUF_H