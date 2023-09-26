# Embedding.cpp

[embedding.cpp](https://github.com/FFengIll/embedding.cpp) is a text embedding tool via `BERT` base model upon [ggml](https://github.com/ggerganov/ggml).

[embedding.cpp](https://github.com/FFengIll/embedding.cpp) is a fork from [bert.cpp](https://github.com/skeskinen/bert.cpp).

> Thanks to `bert.cpp` contributors. Here is also an original [README](./README.origin.md).


> This fork is a result from [pr 32](https://github.com/skeskinen/bert.cpp/pull/32), [pr 31](https://github.com/skeskinen/bert.cpp/pull/31) and [issue 36](https://github.com/skeskinen/bert.cpp/issues/36#issuecomment-1731338977).

## Feature (Origin)
* Plain C/C++ implementation without dependencies
* Inherit support for various architectures from ggml (x86 with AVX2, ARM, etc.)
* Choose your model size from 32/16/4 bits per model weigth
* all-MiniLM-L6-v2 with 4bit quantization is only 14MB. Inference RAM usage depends on the length of the input
* Sample cpp server over tcp socket and a python test client
* Benchmarks to validate correctness and speed of inference

## Feature (Improve)
* Build tokenizer with [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp).
  * Can correctly handle asian writing (CJK, and so on).
  * Can process cased/uncased with respect to origin config in `tokenizer.json`.
* Upgrade to use [GGUF](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md) model file format. So it is easy to expand and keep compatible.

> With above, we can run embedding.cpp with more models like [m3e](), [e5]() and so on.

## Limitation
* Only support bert base model for embedding. other architecture like SGPT is not supported.
* Only run on CPU.
* All outputs are mean pooled and normalized.
* Batching support is WIP. 
  * Lack of real batching means that this library is slower than it could be in usecases where you have multiple sentences.

## Usage

### Checkout submodules
```sh
git submodule update --init --recursive
```

### Build
By default, it build both
- the native binaries, like the example server, with static libraries;
- and the dynamic library for usage from e.g. Python.

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cd ..
```

> rust should be installed.
> see [rust](https://www.rust-lang.org/tools/install) 
> or run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`


### Converting models to gguf format
Converting models is similar to llama.cpp. Use models/convert-to-gguf.py to make hf models into either f32 or f16 gguf models. 
Then use ./build/bin/quantize to turn those into Q4_0, 4bit per weight models.

There is also models/run_conversions.sh which creates all 4 versions (f32, f16, Q4_0, Q4_1) at once.
```sh
cd models
# Clone a model from hf
git clone https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
# Run conversions to 4 ggml formats (f32, f16, Q4_0, Q4_1)
sh run_conversions.sh multi-qa-MiniLM-L6-cos-v1
```