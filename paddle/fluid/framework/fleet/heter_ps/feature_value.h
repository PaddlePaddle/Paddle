/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_HETERPS

#include <iostream>

namespace paddle {
namespace framework {
#define MF_DIM 8

typedef uint64_t FeatureKey;

#ifdef PADDLE_WITH_XPU_KP
typedef uint32_t FidKey;
#endif


struct FeatureValue {
  float delta_score;
  float show;
  float clk;
  int slot;
  float lr;
  float lr_g2sum;
  int mf_size;
  float mf[MF_DIM + 1];
  uint64_t cpu_ptr;

  friend std::ostream& operator<<(std::ostream& out, FeatureValue& val) {
    out << "show:" << val.show << " clk:" << val.clk << " slot:" << val.slot
        << " lr:" << val.lr << " mf_size:" << val.mf_size << " mf:";
    for (int i = 0; i < MF_DIM + 1; ++i) {
      if (i == 0) {
        out << val.mf[i];
      } else {
        out << "," << val.mf[i];
      }
    }
    out << " cpu_ptr:" << val.cpu_ptr;
    return out;
  }
};

// If FeaturePushValue struct change, the size of it can't over 64 bytes.
// Otherwise the merge_grad_kernel, sum_fidseq_add_grad_kernel in XPUPS will cause fault.
struct FeaturePushValue {
  float show;
  float clk;
  int slot;
  float lr_g;
  float mf_g[MF_DIM];

  friend std::ostream& operator<<(std::ostream& out, FeaturePushValue& val) {
    out << "show:" << val.show << " clk:" << val.clk << " slot:" << val.slot
        << " lr_g:" << val.lr_g;
    for (int i = 0; i < MF_DIM; ++i) {
      out << " " << val.mf_g[i];
    }
    return out;
  }

  // __device__ __forceinline__ FeaturePushValue
  // operator+(const FeaturePushValue& a) const {
  //  FeaturePushValue out;
  //  out.slot = a.slot;
  //  out.show = a.show + show;
  //  out.clk = a.clk + clk;
  //  out.lr_g = a.lr_g + lr_g;
  //  for (int i = 0; i < MF_DIM; ++i) {
  //    out.mf_g[i] = a.mf_g[i] + mf_g[i];
  //  }
  //  return out;
  // }
};


}  // end namespace framework
}  // end namespace paddle
#endif
