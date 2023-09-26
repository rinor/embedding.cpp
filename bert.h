#ifndef BERT_H
#define BERT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32)
#define BERT_API __declspec(dllexport)
#else
#define BERT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    enum llama_log_level
    {
        LLAMA_LOG_LEVEL_ERROR = 2,
        LLAMA_LOG_LEVEL_WARN = 3,
        LLAMA_LOG_LEVEL_INFO = 4
    };

    struct bert_params
    {
        int32_t n_threads = 6;
        int32_t port = 8080; // server mode port to bind

        const char *model = "models/all-MiniLM-L6-v2/ggml-model-q4_0.bin"; // model path
        const char *prompt = "test prompt";
    };

    BERT_API bool bert_params_parse(int argc, char **argv, bert_params &params);

    struct bert_ctx;

    typedef int32_t bert_vocab_id;

    BERT_API struct bert_ctx *bert_load_from_file(const char *fname);
    BERT_API void bert_free(bert_ctx *ctx);

    // Main api, does both tokenizing and evaluation

    BERT_API void bert_encode(
        struct bert_ctx *ctx,
        int32_t n_threads,
        const char *texts,
        float *embeddings);

    // n_batch_size - how many to process at a time
    // n_inputs     - total size of texts and embeddings arrays
    BERT_API void bert_encode_batch(
        struct bert_ctx *ctx,
        int32_t n_threads,
        int32_t n_batch_size,
        int32_t n_inputs,
        const char **texts,
        float **embeddings);

    // Api for separate tokenization & eval

    BERT_API void bert_tokenize(
        struct bert_ctx *ctx,
        const char *text,
        bert_vocab_id *tokens,
        int32_t *n_tokens,
        int32_t n_max_tokens);

    BERT_API void bert_eval(
        struct bert_ctx *ctx,
        int32_t n_threads,
        bert_vocab_id *tokens,
        int32_t n_tokens,
        float *embeddings);

    // NOTE: for batch processing the longest input must be first
    BERT_API void bert_eval_batch(
        struct bert_ctx *ctx,
        int32_t n_threads,
        int32_t n_batch_size,
        bert_vocab_id **batch_tokens,
        int32_t *n_tokens,
        float **batch_embeddings);

    BERT_API int32_t bert_n_embd(bert_ctx *ctx);
    BERT_API int32_t bert_n_max_tokens(bert_ctx *ctx);

    BERT_API const char *bert_vocab_id_to_token(bert_ctx *ctx, bert_vocab_id id);

    BERT_API bool bert_model_quantize(const char *fname_inp, const char *fname_out, int ftype);

#ifdef __cplusplus
}
#endif

// model quantization parameters
typedef struct llama_model_quantize_params
{
    int nthread; // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
    // enum llama_ftype ftype;      // quantize to this llama_ftype
    bool allow_requantize;       // allow quantizing non-f32/f16 tensors
    bool quantize_output_tensor; // quantize output.weight
    bool only_copy;              // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
} llama_model_quantize_params;

#endif // BERT_H
