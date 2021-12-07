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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator_kernel_configs.h"

#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#else
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace framework {

using framework::AlgorithmsCache;

// ConvSearchCache using framework::AlgorithmsCache to search
// cudnnConvolutionFwdAlgo_t, cudnnConvolutionBwdDataAlgo_t or
// cudnnConvolutionBwdFilterAlgo_t
class ConvSearchCache {
 public:
  static ConvSearchCache& Instance() {
    static ConvSearchCache instance;
    return instance;
  }
#ifdef PADDLE_WITH_HIP
  AlgorithmsCache<miopenConvFwdAlgorithm_t>* GetForward() {
    return &forward_cache_;
  }
  AlgorithmsCache<miopenConvBwdDataAlgorithm_t>* GetBackwardData() {
    return &backward_data_cache_;
  }
  AlgorithmsCache<miopenConvBwdWeightsAlgorithm_t>* GetBackwardFilter() {
    return &backward_filter_cache_;
  }
  AlgorithmsCache<miopenConvFwdAlgorithm_t>* GetConvFusion() {
    return &fusion_forward_cache_;
  }
#else
  AlgorithmsCache<cudnnConvolutionFwdAlgo_t>* GetForward() {
    return &forward_cache_;
  }
  AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>* GetBackwardData() {
    return &backward_data_cache_;
  }
  AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>* GetBackwardFilter() {
    return &backward_filter_cache_;
  }
  AlgorithmsCache<cudnnConvolutionFwdAlgo_t>* GetConvFusion() {
    return &fusion_forward_cache_;
  }
#endif

 private:
  ConvSearchCache() {}
  ~ConvSearchCache() {}
  ConvSearchCache(const ConvSearchCache&) {}
  ConvSearchCache& operator=(const ConvSearchCache&) {}

#ifdef PADDLE_WITH_HIP
  AlgorithmsCache<miopenConvFwdAlgorithm_t> forward_cache_;
  AlgorithmsCache<miopenConvBwdDataAlgorithm_t> backward_data_cache_;
  AlgorithmsCache<miopenConvBwdWeightsAlgorithm_t> backward_filter_cache_;
  AlgorithmsCache<miopenConvFwdAlgorithm_t> fusion_forward_cache_;
#else
  AlgorithmsCache<cudnnConvolutionFwdAlgo_t> forward_cache_;
  AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t> backward_data_cache_;
  AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t> backward_filter_cache_;
  AlgorithmsCache<cudnnConvolutionFwdAlgo_t> fusion_forward_cache_;
#endif
};

}  // namespace framework
}  // namespace paddle
