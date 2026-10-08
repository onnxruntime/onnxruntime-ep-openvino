// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

///////////////////////////////////////////////////////////////////////////
#define OVEP_TESTS_THROW(fmt, ...)                         \
  {                                                        \
    char input[256];                                       \
    snprintf(input, 256, fmt, ##__VA_ARGS__);              \
                                                           \
    std::stringstream msg;                                 \
    msg << std::endl;                                      \
    msg << "Fatal error in " << __FUNCTION__ << std::endl; \
    msg << "  File: " << __FILE__ << std::endl;            \
    msg << "  Line: " << __LINE__ << std::endl;            \
    msg << "  Message: " << input << std::endl;            \
                                                           \
    throw std::runtime_error(msg.str().c_str());           \
  }
