/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include "baspacho/baspacho/Utils.h"

#ifndef NO_SPRUX_CHECKS
#define SPRUX_CHECKS
#endif  // NO_SPRUX_CHECKS

#define SPRUX_CHECK_WHAT1(a, msg)                 \
  if (!(a)) {                                        \
    ::BaSpaCho::throwError(__FILE__, __LINE__, msg); \
  }

#define SPRUX_CHECK_WHAT2(a, what, v1, v2)                 \
  if (!(a)) {                                                 \
    ::BaSpaCho::throwError(__FILE__, __LINE__, what, v1, v2); \
  }

#if defined(SPRUX_CHECKS) && !defined(__CUDACC__)
#define SPRUX_CHECK(a) SPRUX_CHECK_WHAT1(a, #a)
#define SPRUX_CHECK_OP(a, b, op)                                       \
  {                                                                       \
    auto aEval = a;                                                       \
    auto bEval = b;                                                       \
    SPRUX_CHECK_WHAT2(aEval op bEval, #a " " #op " " #b, aEval, bEval) \
  }
#else
#define SPRUX_CHECK(a) ::BaSpaCho::SPRUX_UNUSED(a)
#define SPRUX_CHECK_OP(a, b, op) ::BaSpaCho::SPRUX_UNUSED(a, b)
#endif

#define SPRUX_CHECK_EQ(a, b) SPRUX_CHECK_OP(a, b, ==)
#define SPRUX_CHECK_LE(a, b) SPRUX_CHECK_OP(a, b, <=)
#define SPRUX_CHECK_LT(a, b) SPRUX_CHECK_OP(a, b, <)
#define SPRUX_CHECK_GE(a, b) SPRUX_CHECK_OP(a, b, >=)
#define SPRUX_CHECK_GT(a, b) SPRUX_CHECK_OP(a, b, >)

#define SPRUX_CHECK_NOTNULL(a)                                       \
  {                                                                     \
    auto aEval = a;                                                     \
    SPRUX_CHECK_WHAT1(aEval != nullptr, "'" #a "' Must be non NULL") \
  }
