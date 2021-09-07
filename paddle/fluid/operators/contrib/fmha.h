/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <vector>

#include "paddle/fluid/operators/contrib/fmha_utils.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
  // The QKV matrices.
  void *qkv_ptr;

  // The stride between rows of the Q, K and V matrices.
  size_t qkv_stride_in_bytes;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_fprop_params : public Qkv_params {
  // The dQKV matrices.
  void *dqkv_ptr;

  // The O matrix (output).
  void *o_ptr;

  // The stride between rows of O.
  int64_t o_stride_in_bytes;

  // The pointer to the S matrix, overwritten by the dP matrix (bwd).
  void *s_ptr;
  // The stride between rows of the S matrix.
  int64_t s_stride_in_bytes;

  // The dimensions.
  int b, h, s, d;

  // The scaling factors for the kernel.
  uint32_t scale_bmm1, scale_softmax, scale_bmm2;

  // array of length b+1 holding starting offset of each sequence.
  int *cu_seqlens;

  // array of length b holding the actual sequence lenghts.
  int *seqlens;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;

  // Scale factor of 1 / (1 - p_dropout).
  float rp_dropout;

  // Scale factor of 1 / (1 - p_dropout), in half2.
  uint32_t scale_dropout;

  // Random state for curand.
  uint64_t seed;

  uint64_t offset;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
