#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

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

struct llama_file
{
    // use FILE * so we don't have to re-open the file to mmap
    FILE *fp;
    size_t size;

    llama_file(const char *fname, const char *mode)
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

    ~llama_file()
    {
        if (fp)
        {
            std::fclose(fp);
        }
    }
};

#endif // LLAMA_H