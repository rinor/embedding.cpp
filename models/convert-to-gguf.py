#!/usr/bin/env python3
# HF bert --> gguf conversion

from __future__ import annotations

import argparse
import itertools
import json
import os
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gguf
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor
from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

if TYPE_CHECKING:
    from typing import TypeAlias

NDArray: TypeAlias = "np.ndarray[Any, Any]"

# reverse HF permute back to original pth layout


def reverse_hf_permute(
    weights: NDArray, n_head: int, n_kv_head: int | None = None
) -> NDArray:
    if n_kv_head is not None and n_head != n_kv_head:
        n_head //= n_kv_head

    return (
        weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
        .swapaxes(1, 2)
        .reshape(weights.shape)
    )


def reverse_hf_permute_part(
    weights: NDArray, n_part: int, n_head: int, n_head_kv: int | None = None
) -> NDArray:
    r = weights.shape[0] // 3
    return reverse_hf_permute(
        weights[r * n_part : r * n_part + r, ...], n_head, n_head_kv
    )


def reverse_hf_part(weights: NDArray, n_part: int) -> NDArray:
    r = weights.shape[0] // 3
    return weights[r * n_part : r * n_part + r, ...]


def count_model_parts(dir_model: str) -> int:
    num_parts = 0

    for filename in os.listdir(dir_model):
        if filename.startswith("pytorch_model-"):
            num_parts += 1

    if num_parts > 0:
        print("gguf: found " + str(num_parts) + " model parts")

    return num_parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace LLaMA model to a GGML compatible file"
    )
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="directory containing model file, or model file itself (*.bin)",
    )
    parser.add_argument(
        "ftype",
        type=int,
        choices=[0, 1],
        default=1,
        nargs="?",
        help="output format - use 0 for float32, 1 for float16",
    )
    return parser.parse_args()


args = parse_args()


class BertConvert:
    def __init__(self, args) -> None:
        self.args = args
        self.dir_model = args.model
        self.ftype = args.ftype

        dir_model = args.model
        ftype = args.ftype
        if not dir_model.is_dir():
            print(f"Error: {args.model} is not a directory", file=sys.stderr)
            sys.exit(1)

        # possible tensor data types
        #   ftype == 0 -> float32
        #   ftype == 1 -> float16

        # map from ftype to string
        ftype_str = ["f32", "f16"]

        if args.outfile is not None:
            fname_out = args.outfile
        else:
            # output in the same directory as the model by default
            fname_out = dir_model / f"ggml-model-{ftype_str[ftype]}.gguf"
        self.fname_out = fname_out

        print("gguf: loading model " + dir_model.name)

        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            hparams = json.load(f)
        print("hello print: ", hparams["architectures"][0])
        if hparams["architectures"][0] != "BertModel":
            print("Model architecture not supported: " + hparams["architectures"][0])

            sys.exit()
        self.hparams = hparams

        # FIXME: use a dummy (invalid) ARCH value
        ARCH = list(gguf.MODEL_ARCH.__members__.values())[-1] + 1
        # NAME= gguf.MODEL_ARCH_NAMES[ARCH]
        NAME = "bert"
        self.gguf_writer = gguf.GGUFWriter(self.fname_out, NAME)

    def convert_hparams(self):
        hparams = self.hparams
        gguf_writer = self.gguf_writer
        dir_model = self.dir_model

        print("gguf: get hparam metadata")

        block_count = hparams["num_hidden_layers"]
        head_count = hparams["num_attention_heads"]

        if "num_key_value_heads" in hparams:
            head_count_kv = hparams["num_key_value_heads"]
        else:
            head_count_kv = head_count

        if "_name_or_path" in hparams:
            hf_repo = hparams["_name_or_path"]
        else:
            hf_repo = ""

        if "max_sequence_length" in hparams:
            ctx_length = hparams["max_sequence_length"]
        elif "max_position_embeddings" in hparams:
            ctx_length = hparams["max_position_embeddings"]
        elif "model_max_length" in hparams:
            ctx_length = hparams["model_max_length"]
        else:
            print("gguf: can not find ctx length parameter.")

            sys.exit()

        gguf_writer.add_name(dir_model.name)
        gguf_writer.add_source_hf_repo(hf_repo)
        gguf_writer.add_tensor_data_layout("")
        gguf_writer.add_context_length(ctx_length)
        gguf_writer.add_embedding_length(hparams["hidden_size"])
        gguf_writer.add_block_count(block_count)
        gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        gguf_writer.add_rope_dimension_count(
            hparams["hidden_size"] // hparams["num_attention_heads"]
        )
        gguf_writer.add_head_count(head_count)
        gguf_writer.add_head_count_kv(head_count_kv)
        gguf_writer.add_layer_norm_eps(hparams["layer_norm_eps"])

        if (
            "rope_scaling" in hparams
            and hparams["rope_scaling"] != None
            and "factor" in hparams["rope_scaling"]
        ):
            if "type" in hparams["rope_scaling"]:
                if hparams["rope_scaling"]["type"] == "linear":
                    gguf_writer.add_rope_scale_linear(hparams["rope_scaling"]["factor"])

    def convert_tokenizer(self):
        hparams = self.hparams
        gguf_writer = self.gguf_writer
        dir_model = self.dir_model

        print("gguf: get tokenizer metadata")

        tokenizer_json_file = dir_model / "tokenizer.json"
        if not tokenizer_json_file.is_file():
            print(f"Error: Missing {tokenizer_json_file}", file=sys.stderr)
            sys.exit(1)

        # FIXME: a special kv to store tokenizer for `tokenizer.cpp`
        with open(dir_model / "tokenizer.json", "rb") as f:
            gguf_writer.add_string("ext.tokenizer.json", f.read())

        tokenizer = AutoTokenizer.from_pretrained(dir_model)

        # simple test
        print(tokenizer.encode("I believe the meaning of life is"))

        # get vocab
        # print(tokenizer.get_vocab())
        vocab = tokenizer.get_vocab()
        if not isinstance(vocab, dict):
            raise TypeError

        # id:key
        reversed_vocab = {idx: key for key, idx in vocab.items()}

        if not os.path.exists(dir_model / "vocab.json"):
            with open(dir_model + "/vocab.json", "w") as f:
                json.dump(reversed_vocab, f, indent=True, ensure_ascii=False)

        # write vocab
        tokens = []
        scores = []
        toktypes = []
        # use vocab_size to confirm size
        for idx in range(hparams["vocab_size"]):
            text = reversed_vocab[idx]
            # print(f"{i}:{text}")
            data = bytes(text, "utf-8")

            tokens.append(data)
            scores.append(0.0)  # dymmy
            toktypes.append(gguf.TokenType.NORMAL)  # dummy

        gguf_writer.add_tokenizer_model("bert")
        gguf_writer.add_token_list(tokens)
        gguf_writer.add_token_scores(scores)
        gguf_writer.add_token_types(toktypes)

        # process special vocab
        special_vocab = gguf.SpecialVocab(dir_model)
        special_vocab.add_to_gguf(gguf_writer)

    def convert_tensor(self):
        hparams = self.hparams
        gguf_writer = self.gguf_writer
        dir_model = self.dir_model
        ftype = self.ftype

        # ggml TENSORS

        # get number of model parts
        num_parts = count_model_parts(dir_model)
        print(f"num_parts:{num_parts}\n")

        self.gguf_writer = gguf_writer
        self.dir_model = dir_model

        # if num_parts == 0:
        #     part_names = iter(("pytorch_model.bin",))
        # else:
        #     part_names = (
        #         f"pytorch_model-{n:05}-of-{num_parts:05}.bin"
        #         for n in range(1, num_parts + 1)
        #     )

        # TODO: load a full model maybe hard, try file process
        model = AutoModel.from_pretrained(dir_model, low_cpu_mem_usage=True)
        print(model)

        list_vars = model.state_dict()
        for name in list_vars.keys():
            print(name, list_vars[name].shape, list_vars[name].dtype)

        # for part_name in part_names:
        #     print("gguf: loading model part '" + part_name + "'")
        #     model_part = torch.load(f"{dir_model}/{part_name}", map_location="cpu")

        for name in list_vars.keys():
            data = list_vars[name]
            if name in [
                "embeddings.position_ids",
                "pooler.dense.weight",
                "pooler.dense.bias",
            ]:
                continue
            print("Processing variable: " + name + " with shape: ", data.shape)

            data = data.squeeze().numpy()

            n_dims = len(data.shape)
            old_dtype = data.dtype
            data_dtype = data.dtype

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            # if ftype == 1 and data_dtype == np.float16 and n_dims == 1:
            #     data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if (
                ftype == 1
                and data_dtype == np.float32
                and name.endswith(".weight")
                and n_dims == 2
            ):
                data = data.astype(np.float16)

            # header
            new_name = name

            print(
                name
                + " -> "
                + new_name
                + ", shape = "
                + str(data.shape)
                + ", n_dims = "
                + str(n_dims)
                + ", "
                + str(old_dtype)
                + " --> "
                + str(data.dtype),
            )

            gguf_writer.add_tensor(new_name, data)

    def done(self):
        gguf_writer = self.gguf_writer

        print("gguf: write header")
        gguf_writer.write_header_to_file()
        print("gguf: write metadata")
        gguf_writer.write_kv_data_to_file()
        print("gguf: write tensors")
        gguf_writer.write_tensors_to_file()

        gguf_writer.close()

        print(f"gguf: model successfully exported to '{self.fname_out}'")
        print("")


convert = BertConvert(args)
convert.convert_hparams()
convert.convert_tokenizer()
convert.convert_tensor()
convert.done()
