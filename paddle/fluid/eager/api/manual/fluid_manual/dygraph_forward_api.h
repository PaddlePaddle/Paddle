// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "glog/logging.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/api/all.h"

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
fused_gate_attention_dygraph_function(
    const paddle::experimental::Tensor& Query,
    const paddle::experimental::Tensor& Key,
    const paddle::experimental::Tensor& QueryWeight,
    const paddle::experimental::Tensor& KeyWeight,
    const paddle::experimental::Tensor& ValueWeight,
    const paddle::experimental::Tensor& QKVWeight,
    const paddle::experimental::Tensor& NonbatchedBias,
    const paddle::experimental::Tensor& SrcMask,
    const paddle::experimental::Tensor& GateWeight,
    const paddle::experimental::Tensor& GateBias,
    const paddle::experimental::Tensor& OutLinearWeight,
    const paddle::experimental::Tensor& OutLinearBias,
    const paddle::framework::AttributeMap& attr_map);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
fused_feedforward_dygraph_function(
    const paddle::experimental::Tensor& X,
    const paddle::experimental::Tensor& Dropout1Seed,
    const paddle::experimental::Tensor& Dropout2Seed,
    const paddle::experimental::Tensor& Linear1Weight,
    const paddle::experimental::Tensor& Linear1Bias,
    const paddle::experimental::Tensor& Linear2Weight,
    const paddle::experimental::Tensor& Linear2Bias,
    const paddle::experimental::Tensor& Ln1Scale,
    const paddle::experimental::Tensor& Ln1Bias,
    const paddle::experimental::Tensor& Ln2Scale,
    const paddle::experimental::Tensor& Ln2Bias,
    const paddle::framework::AttributeMap& attr_map);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
fused_attention_dygraph_function(
    const paddle::experimental::Tensor& X,
    const paddle::experimental::Tensor& LnScale,
    const paddle::experimental::Tensor& LnBias,
    const paddle::experimental::Tensor& QKVW,
    const paddle::experimental::Tensor& QKVBias,
    const paddle::experimental::Tensor& CacheKV,
    const paddle::experimental::Tensor& SrcMask,
    const paddle::experimental::Tensor& OutLinearW,
    const paddle::experimental::Tensor& OutLinearBias,
    const paddle::experimental::Tensor& Ln2Scale,
    const paddle::experimental::Tensor& Ln2Bias,
    const paddle::framework::AttributeMap& attr_map);

paddle::experimental::Tensor fused_gemm_epilogue_dygraph_function(
    const paddle::experimental::Tensor& X,
    const paddle::experimental::Tensor& Y,
    const paddle::experimental::Tensor& Bias,
    const paddle::framework::AttributeMap& attr_map);

std::tuple<paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor,
           paddle::experimental::Tensor>
fused_bias_dropout_residual_layer_norm_dygraph_function(
    const paddle::experimental::Tensor& X,
    const paddle::experimental::Tensor& Residual,
    const paddle::experimental::Tensor& Bias,
    const paddle::experimental::Tensor& LnScale,
    const paddle::experimental::Tensor& LnBias,
    const paddle::framework::AttributeMap& attr_map);
