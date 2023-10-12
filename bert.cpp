#include "bert.h"
#include "ggml.h"
#include "gguf.h"
#include "tokenizer.h"

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
#include <unordered_map>
#include <array>
#include <mutex>

// default hparams (all-MiniLM-L6-v2)
struct bert_hparams
{
    int32_t n_vocab = 30522;
    int32_t n_max_tokens = 512;
    int32_t n_embd = 256;
    int32_t n_intermediate = 1536;
    int32_t n_head = 12;
    int32_t n_layer = 6;
    int32_t n_vocab_size = 2;
    int32_t f16 = 0;
    float eps = 1e-12;
};

struct bert_layer
{
    // normalization
    struct ggml_tensor *ln_att_w;
    struct ggml_tensor *ln_att_b;

    struct ggml_tensor *ln_out_w;
    struct ggml_tensor *ln_out_b;

    // attention
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;
};

struct bert_model
{
    bert_hparams hparams;

    // embeddings weights
    struct ggml_tensor *word_embeddings;
    struct ggml_tensor *token_type_embeddings;
    struct ggml_tensor *position_embeddings;
    struct ggml_tensor *ln_e_w;
    struct ggml_tensor *ln_e_b;

    std::vector<bert_layer> layers;

    struct ggml_context *ctx;
    struct gguf_context *gguf;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct bert_buffer
{
    uint8_t *data = NULL;
    size_t size = 0;

    void resize(size_t size)
    {
        delete[] data;
        data = new uint8_t[size];
        this->size = size;
    }

    ~bert_buffer()
    {
        delete[] data;
    }
};

struct bert_vocab
{
    using id = int32_t;
    using token = std::string;
    using ttype = gguf_token_type;

    struct token_data
    {
        token text;
        float score;
        ttype type;
    };

    std::string tokenizer_json;

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data> id_to_token;

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default bert special tokens
    id special_bos_id = 1;
    id special_eos_id = 2;
    id special_unk_id = 0;
    id special_sep_id = -1; // init with -1 to make it easy to check, default in bert is 102
    id special_pad_id = -1; // default in bert is 0
    id special_cls_id = -1; // default in bert is 101
};

struct bert_ctx
{
    bert_model model;
    bert_vocab vocab;
    bert_tokenizer tokenizer;

    size_t mem_per_token;
    int64_t mem_per_input;
    int32_t max_batch_n;
    bert_buffer buf_compute;
};

//
// Loading and setup
//

struct bert_loader
{
    int n_kv = 0;
    int n_tensors = 0;
    int n_created = 0;

    int64_t n_elements = 0;
    size_t n_bytes = 0;

    bool use_mmap = false;

    ggml_type ftype = ggml_type::GGML_TYPE_COUNT;
    gguf_file file;
    gguf_fver fver;

    struct gguf_context *ctx_gguf = NULL;
    struct ggml_context *ctx_meta = NULL;

    ~bert_loader()
    {
        if (ctx_gguf)
        {
            gguf_free(ctx_gguf);
        }
        if (ctx_meta)
        {
            ggml_free(ctx_meta);
        }
    }

    bert_loader(const char *fname) : file(fname, "rb")
    {

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx_meta,
        };

        ctx_gguf = gguf_init_from_file(fname, params);
        if (!ctx_gguf)
        {
            throw std::runtime_error(format("%s: failed to load model from %s\n", __func__, fname));
        }

        n_kv = gguf_get_n_kv(ctx_gguf);
        n_tensors = gguf_get_n_tensors(ctx_gguf);
        fver = (gguf_fver)gguf_get_version(ctx_gguf);

        for (int i = 0; i < n_tensors; i++)
        {
            const char *name = gguf_get_tensor_name(ctx_gguf, i);
            struct ggml_tensor *t = ggml_get_tensor(ctx_meta, name);
            if (t == NULL)
            {
                throw std::runtime_error(format("%s: can not get tensor %s\n", __func__, name));
            }
            n_elements += ggml_nelements(t);
            n_bytes += ggml_nbytes(t);
        }

        printf("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
               __func__, n_kv, n_tensors, fname, gguf_file_version_name(fver));

        // determine file type based on the number of tensors for each quantization and print meta data
        // TODO: make optional
        {
            std::map<enum ggml_type, uint32_t> n_type;

            uint32_t n_type_max = 0;
            enum ggml_type type_max = GGML_TYPE_F32;

            for (int i = 0; i < n_tensors; i++)
            {
                const char *name = gguf_get_tensor_name(ctx_gguf, i);
                struct ggml_tensor *meta = ggml_get_tensor(ctx_meta, name);

                n_type[meta->type]++;

                if (n_type_max < n_type[meta->type])
                {
                    n_type_max = n_type[meta->type];
                    type_max = meta->type;
                }

                printf("%s: - tensor %4d: %32s %-8s [ %s ]\n", __func__, i, name, ggml_type_name(meta->type), format_tensor_shape(meta).c_str());
            }

            switch (type_max)
            {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
                // case GGML_TYPE_Q5_0:
                // case GGML_TYPE_Q5_1:
                // case GGML_TYPE_Q8_0:
                // case GGML_TYPE_Q2_K:
                // case GGML_TYPE_Q3_K:
                // case GGML_TYPE_Q4_K:
                // case GGML_TYPE_Q5_K:
                // case GGML_TYPE_Q6_K:
                ftype = type_max;
                break;
            default:
            {
                //  LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                ftype = ggml_type::GGML_TYPE_COUNT;
            }
            break;
            }

            // this is a way to mark that we have "guessed" the file type
            // ftype = (llama_ftype)(ftype | LLAMA_FTYPE_GUESSED);

            // {
            //     const int kid = gguf_find_key(ctx_gguf, "general.file_type");
            //     if (kid >= 0)
            //     {
            //         ftype = (llama_ftype)gguf_get_val_u32(ctx_gguf, kid);
            //     }
            // }

            for (int i = 0; i < n_kv; i++)
            {
                const char *name = gguf_get_key(ctx_gguf, i);
                const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

                printf("%s: - kv %3d: %42s %-8s\n", __func__, i, name, gguf_type_name(type));
            }

            // print type counts
            for (auto &kv : n_type)
            {
                if (kv.second == 0)
                {
                    continue;
                }

                printf("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
            }
        }

        // if (!llama_mmap::SUPPORTED) {
        //     LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        //     use_mmap = false;
        // }

        // this->use_mmap = use_mmap;
    }

    const char *get_tensor_name(int i) const
    {
        return gguf_get_tensor_name(ctx_gguf, i);
    }

    struct ggml_tensor *get_tensor_meta(int i) const
    {
        return ggml_get_tensor(ctx_meta, get_tensor_name(i));
    }

    void calc_sizes(bert_model &model, size_t &ctx_size_p, size_t &mmapped_size_p) const
    {
        // ctx_size_p = 0;
        // mmapped_size_p = 0;

        // for (int i = 0; i < n_tensors; i++)
        // {
        //     struct ggml_tensor *meta = get_tensor_meta(i);
        //     ctx_size_p += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
        //     (use_mmap ? mmapped_size_p : ctx_size_p) += ggml_nbytes_pad(meta);
        // }

        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_intermediate = hparams.n_intermediate;
        const int n_vocab = hparams.n_vocab;

        // Calculate size requirements

        mmapped_size_p += n_embd * n_vocab * ggml_type_sizef(ftype);      // word_embeddings
        mmapped_size_p += n_embd * 2 * ggml_type_sizef(ftype);            // token_type_embeddings
        mmapped_size_p += n_embd * n_max_tokens * ggml_type_sizef(ftype); // position_embeddings

        mmapped_size_p += 2 * n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_e_*

        mmapped_size_p += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_*

        mmapped_size_p += 4 * n_layer * (n_embd * n_embd * ggml_type_sizef(ftype)); // kqvo weights
        mmapped_size_p += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32));  // kqvo bias

        mmapped_size_p += 2 * n_layer * (n_embd * n_intermediate * ggml_type_sizef(ftype)); // ff_*_w
        mmapped_size_p += n_layer * (n_intermediate * ggml_type_sizef(GGML_TYPE_F32));      // ff_i_b
        mmapped_size_p += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32));              // ff_o_b

        mmapped_size_p += (5 + 16 * n_layer) * 512; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, mmapped_size_p / (1024.0 * 1024.0));
    }

    struct ggml_tensor *create_tensor_for(struct ggml_context *ctx, struct ggml_tensor *meta, ggml_backend_type backend)
    {
        if (backend != GGML_BACKEND_CPU)
        {
            ggml_set_no_alloc(ctx, true);
        }

        struct ggml_tensor *tensor = ggml_dup_tensor(ctx, meta);
        tensor->backend = backend; // TODO: ggml_set_backend
        ggml_set_name(tensor, ggml_get_name(meta));

        if (backend != GGML_BACKEND_CPU)
        {
            ggml_set_no_alloc(ctx, use_mmap);
        }

        n_created++;

        return tensor;
    }

    struct ggml_tensor *create_tensor(struct ggml_context *ctx, const std::string &name, const std::vector<int64_t> &ne, ggml_backend_type backend)
    {
        struct ggml_tensor *cur = ggml_get_tensor(ctx_meta, name.c_str());

        if (cur == NULL)
        {
            throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
        }

        {
            bool is_ok = true;
            for (size_t i = 0; i < ne.size(); ++i)
            {
                if (ne[i] != cur->ne[i])
                {
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok)
            {
                throw std::runtime_error(
                    format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                           __func__, name.c_str(),
                           format_tensor_shape(ne).c_str(),
                           format_tensor_shape(cur).c_str()));
            }
        }

        return create_tensor_for(ctx, cur, backend);
    }

    size_t file_offset(const char *name) const
    {
        const int idx = gguf_find_tensor(ctx_gguf, name);

        if (idx < 0)
        {
            throw std::runtime_error(format("%s: tensor '%s' not found in the file", __func__, name));
        }

        return gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, idx);
    }

    void load_data_for(struct ggml_tensor *cur) const
    {
        const size_t offs = file_offset(ggml_get_name(cur));
        file.seek(offs, SEEK_SET);
        file.read_raw(cur->data, ggml_nbytes(cur));

        // if (use_mmap)
        // {
        //     cur->data = (uint8_t *)mapping->addr + offs;
        // }
        // else
        // {
        // file.seek(offs, SEEK_SET);
        // file.read_raw(cur->data, ggml_nbytes(cur));
        // }
    }

    void load_all_data(struct ggml_context *ctx)
    {
        size_t size_data = 0;
        size_t size_lock = 0;
        size_t size_pref = 0; // prefetch

        for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); i++)
        {
            struct ggml_tensor *cur = ggml_get_tensor(ctx, gguf_get_tensor_name(ctx_gguf, i));
            size_data += ggml_nbytes(cur);
            if (cur->backend == GGML_BACKEND_CPU)
            {
                size_pref += ggml_nbytes(cur);
            }
        }

        // if (use_mmap)
        // {
        //     mapping.reset(new llama_mmap(&file, size_pref, ggml_is_numa()));
        //     if (lmlock)
        //     {
        //         lmlock->init(mapping->addr);
        //     }
        // }

        size_t done_size = 0;
        for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); i++)
        {
            struct ggml_tensor *cur = ggml_get_tensor(ctx, gguf_get_tensor_name(ctx_gguf, i));
            GGML_ASSERT(cur); // unused tensors should have been caught by load_data already

            // allocate temp buffer if not using mmap
            if (!use_mmap && cur->data == NULL)
            {
                GGML_ASSERT(cur->backend != GGML_BACKEND_CPU);
#ifdef GGML_USE_CPU_HBM
                cur->data = (uint8_t *)hbw_malloc(ggml_nbytes(cur));
#else
                cur->data = (uint8_t *)malloc(ggml_nbytes(cur));
#endif
            }

            load_data_for(cur);

            done_size += ggml_nbytes(cur);
        }
    }

    void llm_print_meta(bert_ctx *bert)
    {
        auto &hparams = bert->model.hparams;
        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_max_tokens   = %d\n", __func__, hparams.n_max_tokens);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_intermediate  = %d\n", __func__, hparams.n_intermediate);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_vocab_size = %d\n", __func__, hparams.n_vocab_size);
        printf("%s: f16     = %d\n", __func__, hparams.f16);

        auto &vocab = bert->vocab;
        printf("%s: vocab.special_bos_id  = %d\n", __func__, vocab.special_bos_id);
        printf("%s: vocab.special_eos_id  = %d\n", __func__, vocab.special_eos_id);
        printf("%s: vocab.special_unk_id  = %d\n", __func__, vocab.special_unk_id);
        printf("%s: vocab.special_sep_id  = %d\n", __func__, vocab.special_sep_id);
        printf("%s: vocab.special_pad_id  = %d\n", __func__, vocab.special_pad_id);
        printf("%s: vocab.special_cls_id  = %d\n", __func__, vocab.special_cls_id);
    }

    void llm_load_hparams(bert_ctx *bert, const LLM_KV &kv)
    {
        // auto *ctx = bert->model.gguf;
        auto *ctx = ctx_gguf;
        bert_hparams &hparams = bert->model.hparams;

        // get general kv
        // GGUF_GET_KEY(ctx, model.name, gguf_get_val_str, GGUF_TYPE_STRING, false, kv(LLM_KV_GENERAL_NAME));

        // get hparams kv
        GGUF_GET_KEY(ctx, hparams.n_vocab, gguf_get_arr_n, GGUF_TYPE_ARRAY, true, kv(LLM_KV_TOKENIZER_LIST));
        GGUF_GET_KEY(ctx, hparams.n_max_tokens, gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_CONTEXT_LENGTH));
        GGUF_GET_KEY(ctx, hparams.n_embd, gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_EMBEDDING_LENGTH));
        GGUF_GET_KEY(ctx, hparams.n_intermediate, gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_FEED_FORWARD_LENGTH));
        GGUF_GET_KEY(ctx, hparams.n_head, gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_ATTENTION_HEAD_COUNT));
        GGUF_GET_KEY(ctx, hparams.n_layer, gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_BLOCK_COUNT));
        GGUF_GET_KEY(ctx, hparams.eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, kv(LLM_KV_ATTENTION_LAYERNORM_EPS));
    }

    void llm_load_tokenizer(bert_ctx *bert, const LLM_KV &kv)
    {
        bert_model &model = bert->model;
        bert_vocab &vocab = bert->vocab;
        auto *ctx = ctx_gguf;

        // general
        const int token_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_LIST).c_str());
        if (token_idx == -1)
        {
            throw std::runtime_error("cannot find tokenizer vocab in model file\n");
        }

        const int score_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_SCORES).c_str());
        if (score_idx == -1)
        {
            throw std::runtime_error("cannot find tokenizer scores in model file\n");
        }

        const float *scores = (const float *)gguf_get_arr_data(ctx, score_idx);

        const int toktype_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str());
        if (toktype_idx == -1)
        {
            throw std::runtime_error("cannot find token type list in GGUF file\n");
        }

        const int *toktypes = (const int *)gguf_get_arr_data(ctx, toktype_idx);

        // determine vocab type
        {
            std::string tokenizer_name;

            GGUF_GET_KEY(ctx, tokenizer_name, gguf_get_val_str, GGUF_TYPE_STRING, true, kv(LLM_KV_TOKENIZER_MODEL));
        }

        const uint32_t n_vocab = gguf_get_arr_n(ctx, token_idx);

        vocab.id_to_token.resize(n_vocab);

        for (uint32_t i = 0; i < n_vocab; i++)
        {
            std::string word = gguf_get_arr_str(ctx, token_idx, i);

            vocab.token_to_id[word] = i;

            auto &token_data = vocab.id_to_token[i];
            token_data.text = std::move(word);
            token_data.score = scores[i];
            token_data.type = (gguf_token_type)toktypes[i];
        }

        // special tokens
        GGUF_GET_KEY(ctx, vocab.special_bos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_BOS_ID));
        GGUF_GET_KEY(ctx, vocab.special_eos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_EOS_ID));
        GGUF_GET_KEY(ctx, vocab.special_unk_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_UNK_ID));
        GGUF_GET_KEY(ctx, vocab.special_sep_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_SEP_ID));
        GGUF_GET_KEY(ctx, vocab.special_pad_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_PAD_ID));
        GGUF_GET_KEY(ctx, vocab.special_cls_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_CLS_ID));

        // extra kv process for tokenizers-cpp
        GGUF_GET_KEY(ctx, vocab.tokenizer_json, gguf_get_val_str, GGUF_TYPE_STRING, true, "blob.tokenizer.json");
        bert->tokenizer.load(vocab.tokenizer_json);
    }

    void llm_load_tensors(bert_ctx *bert)
    {
        bert_model &model = bert->model;

        size_t mmapped_size_p = 0;
        size_t tx_size_p = 0;

        calc_sizes(model, tx_size_p, mmapped_size_p);

        // create the ggml context
        {
            struct ggml_init_params params = {
                .mem_size = mmapped_size_p,
                .mem_buffer = NULL,
                .no_alloc = false,
            };

            model.ctx = ggml_init(params);
            if (!model.ctx)
            {
                throw std::runtime_error(format("%s: ggml_init() failed\n", __func__));
            }
        }

        // prepare memory for the weights
        {
            const auto &hparams = model.hparams;
            auto *ctx = model.ctx;

            const int n_embd = hparams.n_embd;
            const int n_layer = hparams.n_layer;
            const int n_intermediate = hparams.n_intermediate;
            const int n_max_tokens = hparams.n_max_tokens;
            const int n_vocab = hparams.n_vocab;
            const int n_vocab_size = hparams.n_vocab_size;

            const ggml_backend_type backend = GGML_BACKEND_CPU;

            size_t ctx_size;
            size_t mmapped_size;

            model.layers.resize(n_layer);

            model.word_embeddings = create_tensor(ctx, "embeddings.word_embeddings.weight", {n_embd, n_vocab}, backend);
            model.token_type_embeddings = create_tensor(ctx, "embeddings.token_type_embeddings.weight", {n_embd, n_vocab_size}, backend);
            model.position_embeddings = create_tensor(ctx, "embeddings.position_embeddings.weight", {n_embd, n_max_tokens}, backend);

            model.ln_e_w = create_tensor(ctx, "embeddings.LayerNorm.weight", {n_embd}, backend);
            model.ln_e_b = create_tensor(ctx, "embeddings.LayerNorm.bias", {n_embd}, backend);

            for (int i = 0; i < n_layer; ++i)
            {
                auto &layer = model.layers[i];

                layer.ln_att_w = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.weight", {n_embd}, backend);
                layer.ln_att_b = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.bias", {n_embd}, backend);
                layer.ln_out_w = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".output.LayerNorm.weight", {n_embd}, backend);
                layer.ln_out_b = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".output.LayerNorm.bias", {n_embd}, backend);

                layer.q_w = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".attention.self.query.weight", {n_embd, n_embd}, backend);
                layer.q_b = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".attention.self.query.bias", {n_embd}, backend);
                layer.k_w = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".attention.self.key.weight", {n_embd, n_embd}, backend);
                layer.k_b = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".attention.self.key.bias", {n_embd}, backend);
                layer.v_w = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".attention.self.value.weight", {n_embd, n_embd}, backend);
                layer.v_b = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".attention.self.value.bias", {n_embd}, backend);
                layer.o_w = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".attention.output.dense.weight", {n_embd, n_embd}, backend);
                layer.o_b = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".attention.output.dense.bias", {n_embd}, backend);

                layer.ff_i_w = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".intermediate.dense.weight", {n_embd, n_intermediate}, backend);
                layer.ff_i_b = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".intermediate.dense.bias", {n_intermediate}, backend);

                layer.ff_o_w = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".output.dense.weight", {n_intermediate, n_embd}, backend);
                layer.ff_o_b = create_tensor(ctx, "encoder.layer." + std::to_string(i) + ".output.dense.bias", {n_embd}, backend);
            }
        }

        // load read weights
        load_all_data(model.ctx);
    }
};

int32_t bert_n_embd(bert_ctx *ctx)
{
    return ctx->model.hparams.n_embd;
}

int32_t bert_n_max_tokens(bert_ctx *ctx)
{
    return ctx->model.hparams.n_max_tokens;
}

const char *bert_vocab_id_to_token(bert_ctx *ctx, bert_vocab_id id)
{
    bert_vocab &vocab = ctx->vocab;
    return vocab.id_to_token.at(id).text.c_str();
}

//
// Cli interface
//

void bert_print_usage(char **argv, const bert_params &params)
{
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  --port p     port to bind in server mode (default: %d)\n", params.port);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model);
    fprintf(stderr, "\n");
}

bool bert_params_parse(int argc, char **argv, bert_params &params)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "-p" || arg == "--prompt")
        {
            params.prompt = argv[++i];
        }
        else if (arg == "--port")
        {
            params.port = std::stoi(argv[++i]);
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-h" || arg == "--help")
        {
            bert_print_usage(argv, params);
            exit(0);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bert_print_usage(argv, params);
            exit(0);
        }
    }

    return true;
}

//
// Tokenizing
//
void bert_tokenize(
    struct bert_ctx *ctx,
    const char *text,
    bert_vocab_id *tokens,
    int32_t *n_tokens,
    int32_t n_max_tokens)
{
    auto &tokenizer = ctx->tokenizer;

    // TODO: add normalization

    // call Encode to turn prompt into token ids
    std::vector<int> ids = tokenizer.encode(text);

    int cls_tok_id = ctx->vocab.special_cls_id;
    int sep_tok_id = ctx->vocab.special_sep_id;
    int pad_tok_id = ctx->vocab.special_pad_id;

    int32_t t = 0;
    tokens[t++] = cls_tok_id;
    for (auto it = ids.begin(); it != ids.end(); it++)
    {
        // since tokenizers-cpp may do some padding (according to tokenizer.json)
        if (*it == pad_tok_id)
        {
            break;
        }
        tokens[t++] = *it;
        if (t >= n_max_tokens)
        {
            break;
        }
    }

    if (t >= n_max_tokens)
    {
        tokens[n_max_tokens - 1] = sep_tok_id;
    }
    else
    {
        tokens[t++] = sep_tok_id;
    }
    *n_tokens = t;
}

struct bert_ctx *
bert_load_from_file(const char *fname)
{
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname);

    auto *loader = new bert_loader(fname);

    bert_ctx *new_bert = new bert_ctx;
    bert_model &model = new_bert->model;
    bert_vocab &vocab = new_bert->vocab;
    const auto kv = LLM_KV(LLM_ARCH_BERT);

    loader->llm_load_hparams(new_bert, kv);
    loader->llm_load_tokenizer(new_bert, kv);

    loader->llm_print_meta(new_bert);

    loader->llm_load_tensors(new_bert);

    printf(" done\n");

    // Calculate space requirements for setting up context buffers later
    {
        bert_vocab_id tokens[] = {0, 1, 2, 3};
        // TODO: We set the initial buffer size to 32MB and hope it's enough. Maybe there is a better way to do this?
        new_bert->buf_compute.resize(32 * 1024 * 1024);
        bert_eval(new_bert, 1, tokens, 4, nullptr);
        new_bert->max_batch_n = 0;

        // TODO: Max tokens should be a param?
        int32_t N = new_bert->model.hparams.n_max_tokens;
        new_bert->mem_per_input = 1.1 * (new_bert->mem_per_token * N); // add 10% to account for ggml object overhead
    }
    printf("%s: mem_per_token %zu KB, mem_per_input %lld MB\n", __func__, new_bert->mem_per_token / (1 << 10), new_bert->mem_per_input / (1 << 20));

    return new_bert;
}

void bert_resize_ctx(bert_ctx *ctx, int32_t new_size)
{
    int64_t buf_size_new = ctx->mem_per_input * new_size;

    // TODO: Max memory should be a param? Now just 1 GB
    int64_t GB = 1 << 30;
    // printf("%s: requested_buf_size %lldMB\n", __func__, buf_size_new / (1 << 20));
    if (buf_size_new > GB)
    {
        int32_t adjusted_new_size = GB / ctx->mem_per_input;
        if (adjusted_new_size < 1)
            adjusted_new_size = 1;
        // printf("%s: requested batch size %d, actual new batch size %d\n", __func__, new_size, adjusted_new_size);
        new_size = adjusted_new_size;
        buf_size_new = ctx->mem_per_input * new_size;
    }
    if (new_size > ctx->max_batch_n)
    {
        ctx->buf_compute.resize(buf_size_new);
        ctx->max_batch_n = new_size;
    }
}

void bert_free(bert_ctx *ctx)
{
    ggml_free(ctx->model.ctx);
    delete ctx;
}

void bert_eval(
    struct bert_ctx *ctx,
    int32_t n_threads,
    bert_vocab_id *tokens,
    int32_t n_tokens,
    float *embeddings)
{
    bert_eval_batch(ctx, n_threads, 1, &tokens, &n_tokens, embeddings ? &embeddings : nullptr);
}

void bert_eval_batch(
    bert_ctx *ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    bert_vocab_id **batch_tokens,
    int32_t *n_tokens,
    float **batch_embeddings)
{
    const bert_model &model = ctx->model;
    bool mem_req_mode = !batch_embeddings;
    // batch_embeddings is nullptr for the initial memory requirements run
    if (!mem_req_mode && n_batch_size > ctx->max_batch_n)
    {
        bert_resize_ctx(ctx, n_batch_size);
        if (n_batch_size > ctx->max_batch_n)
        {
            fprintf(stderr, "%s: tried to increase buffers to batch size %d but failed\n", __func__, n_batch_size);
            return;
        }
    }

    const float eps = model.hparams.eps;

    // TODO: implement real batching
    for (int ba = 0; ba < n_batch_size; ba++)
    {
        const int N = n_tokens[ba];
        const auto &tokens = batch_tokens[ba];

        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_head = hparams.n_head;

        const int d_head = n_embd / n_head;

        std::vector<float> result;
        if (N > n_max_tokens)
        {
            fprintf(stderr, "Too many tokens, maximum is %d\n", n_max_tokens);
            return;
        }

        auto &mem_per_token = ctx->mem_per_token;
        auto &buf_compute = ctx->buf_compute;

        struct ggml_init_params params = {
            .mem_size = buf_compute.size,
            .mem_buffer = buf_compute.data,
            .no_alloc = false,
        };

        struct ggml_context *ctx0 = ggml_init(params);
        struct ggml_cgraph gf = {};

        // Embeddings. word_embeddings + token_type_embeddings + position_embeddings
        // in bert, it is
        // token_embedding + segment_embedding + position_embedding
        struct ggml_tensor *token_layer = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        memcpy(token_layer->data, tokens, N * ggml_element_size(token_layer));

        struct ggml_tensor *token_types = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        ggml_set_zero(token_types);

        struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        for (int i = 0; i < N; i++)
        {
            ggml_set_i32_1d(positions, i, i);
        }

        struct ggml_tensor *inpL = ggml_get_rows(ctx0, model.word_embeddings, token_layer);

        inpL = ggml_add(ctx0,
                        ggml_get_rows(ctx0, model.token_type_embeddings, token_types),
                        inpL);
        inpL = ggml_add(ctx0,
                        ggml_get_rows(ctx0, model.position_embeddings, positions),
                        inpL);

        // embd norm
        {
            inpL = ggml_norm(ctx0, inpL, eps);

            inpL = ggml_add(ctx0,
                            ggml_mul(ctx0,
                                     ggml_repeat(ctx0, model.ln_e_w, inpL),
                                     inpL),
                            ggml_repeat(ctx0, model.ln_e_b, inpL));
        }
        // layers
        for (int il = 0; il < n_layer; il++)
        {
            struct ggml_tensor *cur = inpL;

            // self-attention (multiple head)
            {
                // linear
                struct ggml_tensor *Qcur = cur;
                Qcur = ggml_reshape_3d(ctx0,
                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, Qcur),
                                                ggml_mul_mat(ctx0, model.layers[il].q_w, Qcur)),
                                       d_head, n_head, N);
                struct ggml_tensor *Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

                struct ggml_tensor *Kcur = cur;
                Kcur = ggml_reshape_3d(ctx0,
                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, Kcur),
                                                ggml_mul_mat(ctx0, model.layers[il].k_w, Kcur)),
                                       d_head, n_head, N);
                struct ggml_tensor *K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);

                struct ggml_tensor *Vcur = cur;
                Vcur = ggml_reshape_3d(ctx0,
                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, Vcur),
                                                ggml_mul_mat(ctx0, model.layers[il].v_w, Vcur)),
                                       d_head, n_head, N);
                struct ggml_tensor *V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);

                // Scaled Dot-Product Attention
                // KQ = soft_max(KQ / sqrt(head width))
                struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);
                KQ = ggml_soft_max(ctx0,
                                   ggml_scale(ctx0,
                                              KQ,
                                              ggml_new_f32(ctx0, 1.0f / sqrt((float)d_head))));

                V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
                struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ);
                KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

                cur = ggml_cpy(ctx0,
                               KQV,
                               ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
            }
            // attention output
            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].o_b, cur),
                           ggml_mul_mat(ctx0, model.layers[il].o_w, cur));

            // Add & Norm
            // re-add the layer input
            cur = ggml_add(ctx0, cur, inpL);

            // attention norm
            {
                cur = ggml_norm(ctx0, cur, eps);

                cur = ggml_add(ctx0,
                               ggml_mul(ctx0,
                                        ggml_repeat(ctx0, model.layers[il].ln_att_w, cur),
                                        cur),
                               ggml_repeat(ctx0, model.layers[il].ln_att_b, cur));
            }
            struct ggml_tensor *att_output = cur;

            // Forward Feed
            // intermediate_output = self.intermediate(attention_output)
            cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].ff_i_b, cur),
                           cur);
            cur = ggml_gelu(ctx0, cur);

            // layer_output = self.output(intermediate_output, attention_output)
            cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].ff_o_b, cur),
                           cur);

            // Add & Norm
            // attentions bypass the intermediate layer
            cur = ggml_add(ctx0, att_output, cur);

            // output norm
            {
                cur = ggml_norm(ctx0, cur, eps);

                cur = ggml_add(ctx0,
                               ggml_mul(ctx0,
                                        ggml_repeat(ctx0, model.layers[il].ln_out_w, cur),
                                        cur),
                               ggml_repeat(ctx0, model.layers[il].ln_out_b, cur));
            }
            inpL = cur;
        }
        inpL = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));

        // pooling
        // FIXME: pooling method is hard code here
        struct ggml_tensor *sum = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N, 1);
        ggml_set_f32(sum, 1.0f / N);
        inpL = ggml_mul_mat(ctx0, inpL, sum);

        // normalizer
        ggml_tensor *length = ggml_sqrt(ctx0,
                                        ggml_sum(ctx0, ggml_sqr(ctx0, inpL)));
        inpL = ggml_scale(ctx0, inpL, ggml_div(ctx0, ggml_new_f32(ctx0, 1.0f), length));

        ggml_tensor *output = inpL;
        // run the computation
        ggml_build_forward_expand(&gf, output);
        ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

        // float *dat = ggml_get_data_f32(output);
        // pretty_print_tensor(dat, output->ne, output->nb, output->n_dims - 1, "");

#ifdef GGML_PERF
        // print timing information per ggml operation (for debugging purposes)
        // requires GGML_PERF to be defined
        ggml_graph_print(&gf);
#endif

        if (!mem_req_mode)
        {
            memcpy(batch_embeddings[ba], (float *)ggml_get_data(output), sizeof(float) * n_embd);
        }
        else
        {
            mem_per_token = ggml_used_mem(ctx0) / N;

            printf("used_mem = %zu KB \n", ggml_used_mem(ctx0) / 1024);
            printf("mem_per_token = %zu KB \n", mem_per_token / 1024);
        }

        ggml_free(ctx0);
    }
}

void bert_encode(
    struct bert_ctx *ctx,
    int32_t n_threads,
    const char *texts,
    float *embeddings)
{
    bert_encode_batch(ctx, n_threads, 1, 1, &texts, &embeddings);
}

void bert_encode_batch(
    struct bert_ctx *ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    int32_t n_inputs,
    const char **texts,
    float **embeddings)
{
    // TODO: Disable batching for now
    n_batch_size = 1;
    /*
    if (n_batch_size > n_inputs) {
        n_batch_size = n_inputs;
    }
    if (n_batch_size > ctx->max_batch_n) {
        bert_resize_ctx(ctx, n_batch_size);
        n_batch_size = ctx->max_batch_n;
    }
    */

    int32_t N = bert_n_max_tokens(ctx);

    std::vector<bert_vocab_id> buf_tokens;
    // Most of this buffer will be unused in typical case where inputs are not that long.
    buf_tokens.resize(N * n_inputs);
    std::vector<int32_t> n_tokens = std::vector<int32_t>(n_inputs);
    std::vector<bert_vocab_id *> unsorted_tokens(n_inputs);
    bert_vocab_id *ids = buf_tokens.data();

    for (int i = 0; i < n_inputs; i++)
    {
        unsorted_tokens[i] = ids;

        bert_tokenize(ctx, texts[i], ids, &n_tokens[i], N);

        ids += n_tokens[i];
    }

    if (n_batch_size == n_inputs)
    {
        bert_eval_batch(ctx, n_threads, n_batch_size, unsorted_tokens.data(), n_tokens.data(), embeddings);
    }
    else
    {
        // sort the inputs by tokenized length, batch and eval

        std::vector<int> indices;
        indices.reserve(n_inputs);
        for (int i = 0; i < n_inputs; i++)
        {
            indices.push_back(i);
        }

        std::vector<int32_t> sorted_n_tokens = std::vector<int32_t>(n_inputs);

        std::vector<bert_vocab_id *> sorted_tokens(n_inputs);

        std::sort(indices.begin(), indices.end(), [&](int a, int b)
                  { return n_tokens[a] < n_tokens[b]; });

        std::vector<float *> sorted_embeddings(n_inputs);
        memcpy(sorted_embeddings.data(), embeddings, n_inputs * sizeof(float *));

        for (int i = 0; i < n_inputs; i++)
        {
            sorted_embeddings[i] = embeddings[indices[i]];
            sorted_tokens[i] = unsorted_tokens[indices[i]];
            sorted_n_tokens[i] = n_tokens[indices[i]];
        }

        for (int i = 0; i < n_inputs; i += n_batch_size)
        {
            if (i + n_batch_size > n_inputs)
            {
                n_batch_size = n_inputs - i;
            }
            bert_eval_batch(ctx, n_threads, n_batch_size, &sorted_tokens[i], &sorted_n_tokens[i], &sorted_embeddings[i]);
        }
    }
}

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

//
// Quantize
//

template <typename T>
struct no_init
{
    T value;
    no_init()
    { /* do nothing */
    }
};

static void zeros(std::ofstream &file, size_t n)
{
    char zero = 0;
    for (size_t i = 0; i < n; ++i)
    {
        file.write(&zero, 1);
    }
}

static void llama_convert_tensor_internal(
    struct ggml_tensor *tensor, std::vector<no_init<float>> &output, std::vector<std::thread> &workers,
    const size_t nelements, const int nthread)
{
    if (output.size() < nelements)
    {
        output.resize(nelements);
    }
    float *f32_output = (float *)output.data();

    ggml_type_traits_t qtype;
    if (ggml_is_quantized(tensor->type))
    {
        qtype = ggml_internal_get_type_traits(tensor->type);
        if (qtype.to_float == NULL)
        {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    }
    else if (tensor->type != GGML_TYPE_F16)
    {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2)
    {
        if (tensor->type == GGML_TYPE_F16)
        {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        }
        else if (ggml_is_quantized(tensor->type))
        {
            qtype.to_float(tensor->data, f32_output, nelements);
        }
        else
        {
            GGML_ASSERT(false); // unreachable
        }
        return;
    }

    auto block_size = tensor->type == GGML_TYPE_F16 ? 1 : (size_t)ggml_blck_size(tensor->type);
    auto block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    auto nblocks = nelements / block_size;
    auto blocks_per_thread = nblocks / nthread;
    auto spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    for (auto tnum = 0, in_buff_offs = 0, out_buff_offs = 0; tnum < nthread; tnum++)
    {
        auto thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        auto thr_elems = thr_blocks * block_size;                                       // number of elements for this thread
        auto thr_block_bytes = thr_blocks * block_size_bytes;                           // number of input bytes for this thread

        auto compute = [qtype](ggml_type typ, uint8_t *inbuf, float *outbuf, int nels)
        {
            if (typ == GGML_TYPE_F16)
            {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            }
            else
            {
                qtype.to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *)tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto &w : workers)
    {
        w.join();
    }
    workers.clear();
}

bool bert_model_quantize(const char *fname_inp, const char *fname_out, int ftype_out)
{
    ggml_type quantized_type = GGML_TYPE_Q4_1;

    switch (ftype_out)
    {
    case 2:
        quantized_type = GGML_TYPE_Q4_0;
        break;
    case 3:
        quantized_type = GGML_TYPE_Q4_1;
        break;
    default:
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, ftype_out);
        return false;
    };

    if (quantized_type != GGML_TYPE_Q4_0 && quantized_type != GGML_TYPE_Q4_1)
    {
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, quantized_type);
        return false;
    }

    printf("%s: loading model from '%s'\n", __func__, fname_inp);

    auto *loader = new bert_loader(fname_inp);

    bert_ctx *new_bert = new bert_ctx;
    bert_model &model = new_bert->model;
    bert_vocab &vocab = new_bert->vocab;
    const auto kv = LLM_KV(LLM_ARCH_BERT);

    loader->llm_load_hparams(new_bert, kv);
    loader->llm_load_tokenizer(new_bert, kv);

    loader->llm_print_meta(new_bert);

    loader->llm_load_tensors(new_bert);

    printf(" done\n");

    llama_model_quantize_params *params = new llama_model_quantize_params;
    params->nthread = 0;
    params->only_copy = false;
    params->quantize_output_tensor = quantized_type;

    // MENTION: below is a copy fork from llama.cpp
    int nthread = params->nthread;

    if (nthread <= 0)
    {
        nthread = std::thread::hardware_concurrency();
    }

    if (params->only_copy)
    {
        ftype_out = loader->ftype;
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    struct gguf_context *ctx_out = gguf_init_empty();

    // copy the KV pairs from the input file
    gguf_set_kv(ctx_out, loader->ctx_gguf);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", ftype_out);

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<int64_t> hist_all(1 << 4, 0);

    std::vector<std::thread> workers;
    workers.reserve(nthread);
    std::mutex mutex;

    int idx = 0;

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    // populate the original tensors so we get an initial meta data
    for (int i = 0; i < loader->n_tensors; ++i)
    {
        struct ggml_tensor *meta = loader->get_tensor_meta(i);
        gguf_add_tensor(ctx_out, meta);
    }

    std::ofstream fout(fname_out, std::ios::binary);

    const size_t meta_size = gguf_get_meta_size(ctx_out);

    printf("%s: meta size = %zu bytes\n", __func__, meta_size);

    // placeholder for the meta data
    ::zeros(fout, meta_size);

    for (int i = 0; i < loader->n_tensors; ++i)
    {
        struct ggml_tensor *tensor = loader->get_tensor_meta(i);

        const std::string name = ggml_get_name(tensor);

        if (read_data.size() < ggml_nbytes(tensor))
        {
            read_data.resize(ggml_nbytes(tensor));
        }
        tensor->data = read_data.data();
        loader->load_data_for(tensor);

        printf("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx, loader->n_tensors,
               ggml_get_name(tensor),
               "",
               format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

        // quantize only 2D tensors
        quantize &= (tensor->n_dims == 2);
        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= !params->only_copy;

        enum ggml_type new_type;
        void *new_data;
        size_t new_size;

        if (quantize)
        {
            new_type = quantized_type;

            // If we've decided to quantize to the same type the tensor is already
            // in then there's nothing to do.
            quantize = tensor->type != new_type;
        }
        if (!quantize)
        {
            new_type = tensor->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor);
            printf("size = %8.3f MB\n", ggml_nbytes(tensor) / 1024.0 / 1024.0);
        }
        else
        {
            const size_t nelements = ggml_nelements(tensor);

            float *f32_data;

            if (tensor->type == GGML_TYPE_F32)
            {
                f32_data = (float *)tensor->data;
            }
            else if (ggml_is_quantized(tensor->type) && !params->allow_requantize)
            {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            }
            else
            {
                llama_convert_tensor_internal(tensor, f32_conv_buf, workers, nelements, nthread);
                f32_data = (float *)f32_conv_buf.data();
            }

            printf("quantizing to %s .. ", ggml_type_name(new_type));
            fflush(stdout);

            if (work.size() < nelements * 4)
            {
                work.resize(nelements * 4); // upper bound on size
            }
            new_data = work.data();
            std::array<int64_t, 1 << 4> hist_cur = {};

            static const int chunk_size = 32 * 512;
            const int nchunk = (nelements + chunk_size - 1) / chunk_size;
            const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
            if (nthread_use < 2)
            {
                new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nelements, hist_cur.data());
            }
            else
            {
                size_t counter = 0;
                new_size = 0;
                auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, nelements]()
                {
                    std::array<int64_t, 1 << 4> local_hist = {};
                    size_t local_size = 0;
                    while (true)
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        size_t first = counter;
                        counter += chunk_size;
                        if (first >= nelements)
                        {
                            if (local_size > 0)
                            {
                                for (int j = 0; j < int(local_hist.size()); ++j)
                                {
                                    hist_cur[j] += local_hist[j];
                                }
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        size_t last = std::min(nelements, first + chunk_size);
                        local_size += ggml_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
                    }
                };
                for (int it = 0; it < nthread_use - 1; ++it)
                {
                    workers.emplace_back(compute);
                }
                compute();
                for (auto &w : workers)
                {
                    w.join();
                }
                workers.clear();
            }

            printf("size = %8.2f MB -> %8.2f MB | hist: ", ggml_nbytes(tensor) / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
            int64_t tot_count = 0;
            for (size_t i = 0; i < hist_cur.size(); i++)
            {
                hist_all[i] += hist_cur[i];
                tot_count += hist_cur[i];
            }

            if (tot_count > 0)
            {
                for (size_t i = 0; i < hist_cur.size(); i++)
                {
                    printf("%5.3f ", hist_cur[i] / float(nelements));
                }
            }
            printf("\n");
        }
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;

        // update the gguf meta data as we go
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data, new_size);

        // write tensor data + padding
        fout.write((const char *)new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }

    // go back to beginning of file and write the updated meta data
    {
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *)data.data(), data.size());
    }

    fout.close();

    gguf_free(ctx_out);

    printf("%s: model size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
    printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);

    // print histogram for all tensors
    {
        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); i++)
        {
            sum_all += hist_all[i];
        }

        if (sum_all > 0)
        {
            printf("%s: hist: ", __func__);
            for (size_t i = 0; i < hist_all.size(); i++)
            {
                printf("%5.3f ", hist_all[i] / float(sum_all));
            }
            printf("\n");
        }
    }
    return true;
}
