// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "ck/ck.hpp"
#include "profile.h"

namespace ap {

constexpr int kNumConfigsHalf = 20;
constexpr int kNumConfigsFloat = 0;

#define AP_AUTOTUNE_half(func, stream_ptr, ...)  AP_AUTOTUNE(func, stream_ptr, ap::kNumConfigsHalf, __VA_ARGS__)
// #define AP_AUTOTUNE_float(func, stream, ...)  AP_AUTOTUNE(func, kNumConfigsFloat, stream, __VA_ARGS__)


template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

template <typename ElementT, int Id = 0>
struct GemmTuningConfigs {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 64;
  static constexpr int kNPerBlock = 64;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<8,2>;
  using NThreadCluster = S<8,2>;

  using ABlockTransferThreadSliceLengths = S<2,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,64,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,32,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};


template <typename ElementT>
struct GemmTuningConfigs<ElementT, 1> {
  static constexpr int kBlockSize = 128;
  static constexpr int kMPerBlock = 16;
  static constexpr int kNPerBlock = 32;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 1;
  static constexpr int kN1PerThreadN111 = 1;

  using MThreadCluster = S<4,2>;
  using NThreadCluster = S<2,8>;

  using ABlockTransferThreadSliceLengths = S<1,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<8,1,16,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,16,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 1;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 2> {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 32;
  static constexpr int kNPerBlock = 64;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 1;

  using MThreadCluster = S<2,4>;
  using NThreadCluster = S<4,8>;

  using ABlockTransferThreadSliceLengths = S<1,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<8,1,32,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,32,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 1;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 3> {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 64;
  static constexpr int kNPerBlock = 64;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<4,4>;
  using NThreadCluster = S<2,8>;

  using ABlockTransferThreadSliceLengths = S<2,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,64,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,32,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 4> {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 64;
  static constexpr int kNPerBlock = 128;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 4;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<2,4>;
  using NThreadCluster = S<8,4>;

  using ABlockTransferThreadSliceLengths = S<2,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,64,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<2,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<4,1,64,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 5> {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 32;
  static constexpr int kNPerBlock = 128;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<2,4>;
  using NThreadCluster = S<4,8>;

  using ABlockTransferThreadSliceLengths = S<1,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<8,1,32,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<2,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<4,1,64,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 6> {
  static constexpr int kBlockSize = 128;
  static constexpr int kMPerBlock = 32;
  static constexpr int kNPerBlock = 64;
  static constexpr int kK0PerBlock = 4;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<2,4>;
  using NThreadCluster = S<2,8>;

  using ABlockTransferThreadSliceLengths = S<1,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,32,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<4,1,32,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 7> {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 64;
  static constexpr int kNPerBlock = 64;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<8,2>;
  using NThreadCluster = S<2,8>;

  using ABlockTransferThreadSliceLengths = S<2,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,64,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,32,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 8> {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 64;
  static constexpr int kNPerBlock = 64;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<4,4>;
  using NThreadCluster = S<4,4>;

  using ABlockTransferThreadSliceLengths = S<2,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,64,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,32,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 9> {
  static constexpr int kBlockSize = 128;
  static constexpr int kMPerBlock = 32;
  static constexpr int kNPerBlock = 32;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 1;

  using MThreadCluster = S<2,4>;
  using NThreadCluster = S<2,8>;

  using ABlockTransferThreadSliceLengths = S<2,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,32,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,16,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 1;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 10> {
  static constexpr int kBlockSize = 128;
  static constexpr int kMPerBlock = 64;
  static constexpr int kNPerBlock = 32;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<2,8>;
  using NThreadCluster = S<2,4>;

  using ABlockTransferThreadSliceLengths = S<4,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<2,1,64,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,16,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 11> {
  static constexpr int kBlockSize = 128;
  static constexpr int kMPerBlock = 16;
  static constexpr int kNPerBlock = 32;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 1;
  static constexpr int kN1PerThreadN111 = 1;

  using MThreadCluster = S<4,2>;
  using NThreadCluster = S<8,2>;

  using ABlockTransferThreadSliceLengths = S<1,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<8,1,16,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,16,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 1;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 12> {
  static constexpr int kBlockSize = 128;
  static constexpr int kMPerBlock = 16;
  static constexpr int kNPerBlock = 32;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 1;
  static constexpr int kN1PerThreadN111 = 1;

  using MThreadCluster = S<2,4>;
  using NThreadCluster = S<2,8>;

  using ABlockTransferThreadSliceLengths = S<1,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<8,1,16,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,16,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 1;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 13> {
  static constexpr int kBlockSize = 128;
  static constexpr int kMPerBlock = 32;
  static constexpr int kNPerBlock = 32;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 1;

  using MThreadCluster = S<2,4>;
  using NThreadCluster = S<4,4>;

  using ABlockTransferThreadSliceLengths = S<2,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,32,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,16,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 1;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 14> {
  static constexpr int kBlockSize = 128;
  static constexpr int kMPerBlock = 64;
  static constexpr int kNPerBlock = 32;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<8,2>;
  using NThreadCluster = S<2,4>;

  using ABlockTransferThreadSliceLengths = S<4,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<2,1,64,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,16,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 15> {
  static constexpr int kBlockSize = 128;
  static constexpr int kMPerBlock = 64;
  static constexpr int kNPerBlock = 32;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<8,2>;
  using NThreadCluster = S<2,4>;

  using ABlockTransferThreadSliceLengths = S<2,1,2,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,32,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,16,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 16> {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 64;
  static constexpr int kNPerBlock = 64;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 2;
  static constexpr int kN1PerThreadN111 = 2;

  using MThreadCluster = S<2,8>;
  using NThreadCluster = S<2,8>;

  using ABlockTransferThreadSliceLengths = S<2,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,64,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,32,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 2;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 17> {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 128;
  static constexpr int kNPerBlock = 128;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 4;
  static constexpr int kN1PerThreadN111 = 4;

  using MThreadCluster = S<2,8>;
  using NThreadCluster = S<2,8>;

  using ABlockTransferThreadSliceLengths = S<4,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<2,1,128,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<2,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<4,1,64,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 4;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 18> {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 128;
  static constexpr int kNPerBlock = 128;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 4;
  static constexpr int kN1PerThreadN111 = 4;

  using MThreadCluster = S<8,2>;
  using NThreadCluster = S<8,2>;

  using ABlockTransferThreadSliceLengths = S<4,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<2,1,128,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,4,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,32,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 4;
};
        
template <typename ElementT>
struct GemmTuningConfigs<ElementT, 19> {
  static constexpr int kBlockSize = 256;
  static constexpr int kMPerBlock = 64;
  static constexpr int kNPerBlock = 64;
  static constexpr int kK0PerBlock = 8;
  static constexpr int kK1 = 4;
  static constexpr int kM1PerThreadM111 = 4;
  static constexpr int kN1PerThreadN111 = 1;

  using MThreadCluster = S<2,4>;
  using NThreadCluster = S<4,8>;

  using ABlockTransferThreadSliceLengths = S<2,1,1,4>;
  using ABlockTransferThreadClusterLengths = S<4,1,64,1>;
  using ABlockTransferSrcVectorTensorLengths = S<1,1,1,4>;
  using ABlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  using BBlockTransferThreadSliceLengths = S<1,1,2,4>;
  using BBlockTransferThreadClusterLengths = S<8,1,32,1>;
  using BBlockTransferSrcVectorTensorLengths = S<1,1,2,1>;
  using BBlockTransferDstVectorTensorLengths = S<1,1,1,4>;

  static constexpr int CThreadTransferDstScalarPerVector = 1;
};
        
struct DefaultConfig {
    static constexpr int kConfigId = 7;
};

} // namespace ap

