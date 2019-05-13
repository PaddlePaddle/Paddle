/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/cudnn_helper.h"

DECLARE_uint64(conv_workspace_size_limit);
DECLARE_bool(cudnn_exhaustive_search);
DECLARE_int64(cudnn_exhaustive_search_times);

namespace paddle {
namespace operators {

static constexpr char kCUDNNFwdAlgoCache[] = "kCUDNNFwdAlgoCache";
static constexpr char kCUDNNBwdDataAlgoCache[] = "kCUDNNBwdDataAlgoCache";
static constexpr char kCUDNNBwdFilterAlgoCache[] = "kCUDNNBwdFilterAlgoCache";

#if CUDNN_VERSION_MIN(6, 0, 5)
static constexpr size_t kNUM_CUDNN_FWD_ALGS = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
static constexpr size_t kNUM_CUDNN_BWD_FILTER_ALGS =
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
static constexpr size_t kNUM_CUDNN_BWD_DATA_ALGS =
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
#else
// cuDNN v5 has no CUDNN_CONVOLUTION_FWD_ALGO_COUNT etc.
static constexpr size_t kNUM_CUDNN_FWD_ALGS = 7;
static constexpr size_t kNUM_CUDNN_BWD_FILTER_ALGS = 4;
static constexpr size_t kNUM_CUDNN_BWD_DATA_ALGS = 5;
#endif

}  // namespace operators
}  // namespace paddle
