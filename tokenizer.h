#ifndef BERT_TOKENIZER_H
#define BERT_TOKENIZER_H

#include "tokenizers_cpp.h"

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

using tokenizers::Tokenizer;

struct bert_tokenizer
{
    std::unique_ptr<Tokenizer> tok;

    bert_tokenizer(const std::string &blob);
    ~bert_tokenizer();
    std::string decode(const std::vector<int> &ids);
    std::string decode(const int32_t id);
    std::vector<int> encode(const std::string &text);
    std::string load_bytes_from_file(const std::string &path);
    void print_encode_result(const std::vector<int> &ids);
};

#endif