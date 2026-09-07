// Copyright (c) 2026 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

// xtrans's cublas_v2.h has a "Paddle-specific symbol redirection" block
// guarded by #ifdef PADDLE_WITH_CUDA that textually rewrites several cublas
// function names (e.g. cublasStrsmBatched, cublasGemmStridedBatchedEx) to
// their _v2 counterparts, on the assumption that libcublas.so exports both
// the original int-width symbols and the int64_t-widened _v2 symbols (as
// NVIDIA's official cuBLAS does since some CUDA version). xtrans's own
// libcublas.so only exports the original symbols, so including cublas_v2.h
// with PADDLE_WITH_CUDA defined leaves CINN's CUDA runtime with unresolved
// _v2 references at link time.
//
// Include this header instead of <cublas_v2.h> directly in
// paddle/cinn/runtime/cuda to get the original (non-_v2) symbol names under
// xtrans, while leaving the official NVIDIA CUDA build path untouched.
#if defined(PADDLE_WITH_XPU_CADA) && defined(PADDLE_WITH_CUDA)
#undef PADDLE_WITH_CUDA
#include <cublas_v2.h>
#define PADDLE_WITH_CUDA
#else
#include <cublas_v2.h>
#endif
