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

std::tuple<paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor>
fused_gate_attention_dygraph_function(
    const paddle::Tensor& Query,
    const paddle::Tensor& Key,
    const paddle::Tensor& QueryWeight,
    const paddle::Tensor& KeyWeight,
    const paddle::Tensor& ValueWeight,
    const paddle::Tensor& QKVWeight,
    const paddle::Tensor& NonbatchedBias,
    const paddle::Tensor& SrcMask,
    const paddle::Tensor& GateWeight,
    const paddle::Tensor& GateBias,
    const paddle::Tensor& OutLinearWeight,
    const paddle::Tensor& OutLinearBias,
    const paddle::framework::AttributeMap& attr_map);

std::tuple<paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor>
fused_feedforward_dygraph_function(
    const paddle::Tensor& X,
    const paddle::Tensor& Dropout1Seed,
    const paddle::Tensor& Dropout2Seed,
    const paddle::Tensor& Linear1Weight,
    const paddle::Tensor& Linear1Bias,
    const paddle::Tensor& Linear2Weight,
    const paddle::Tensor& Linear2Bias,
    const paddle::Tensor& Ln1Scale,
    const paddle::Tensor& Ln1Bias,
    const paddle::Tensor& Ln2Scale,
    const paddle::Tensor& Ln2Bias,
    const paddle::framework::AttributeMap& attr_map);

std::tuple<paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor>
fused_attention_dygraph_function(
    const paddle::Tensor& X,
    const paddle::Tensor& LnScale,
    const paddle::Tensor& LnBias,
    const paddle::Tensor& QKVW,
    const paddle::Tensor& QKVBias,
    const paddle::Tensor& CacheKV,
    const paddle::Tensor& SrcMask,
    const paddle::Tensor& OutLinearW,
    const paddle::Tensor& OutLinearBias,
    const paddle::Tensor& Ln2Scale,
    const paddle::Tensor& Ln2Bias,
    const paddle::framework::AttributeMap& attr_map);

paddle::Tensor fused_gemm_epilogue_dygraph_function(
    const paddle::Tensor& X,
    const paddle::Tensor& Y,
    const paddle::Tensor& Bias,
    const paddle::framework::AttributeMap& attr_map);

std::tuple<paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor,
           paddle::Tensor>
fused_bias_dropout_residual_layer_norm_dygraph_function(
    const paddle::Tensor& X,
    const paddle::Tensor& Residual,
    const paddle::Tensor& Bias,
    const paddle::Tensor& LnScale,
    const paddle::Tensor& LnBias,
    const paddle::framework::AttributeMap& attr_map);
