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

#include <NvInfer.h>
#include <string>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/macros.h"
#include "paddle/utils/flat_hash_map.h"

namespace paddle {
namespace inference {
namespace tensorrt {

using SupportsFormateFn = bool (*)(int32_t pos,
                                   const nvinfer1::PluginTensorDesc* inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs);

using AutoGenerateSupportsFormateFn =
    bool (*)(int32_t pos,
             const nvinfer1::PluginTensorDesc* inOut,
             int32_t nbInputs,
             int32_t nbOutputs,
             const paddle::OpMetaInfo& info,
             const framework::OpDesc& desc);

class SupportsFormateFnFactory {
 public:
  static SupportsFormateFnFactory& Instance() {
    static SupportsFormateFnFactory g_meta_fn_map;
    return g_meta_fn_map;
  }

  bool Contains(const std::string& op_name) const {
    return meta_fn_map_.count(op_name) > 0;
  }

  void Insert(std::string op_name, SupportsFormateFn supports_formate_fn) {
    PADDLE_ENFORCE_NE(
        Contains(op_name),
        true,
        phi::errors::AlreadyExists(
            "`%s` op's supportsFormatCombination has been registered.",
            op_name));
    meta_fn_map_.insert({std::move(op_name), supports_formate_fn});
  }

  const SupportsFormateFn& Get(const std::string& op_name) const {
    auto it = meta_fn_map_.find(op_name);
    PADDLE_ENFORCE_NE(
        it,
        meta_fn_map_.end(),
        phi::errors::NotFound(
            "`%s` op's supportsFormatCombination has not been registered.",
            op_name));
    return it->second;
  }

  bool ContainsAuto(const std::string& op_name) const {
    return auto_meta_fn_map_.count(op_name) > 0;
  }

  void InsertAuto(std::string op_name,
                  AutoGenerateSupportsFormateFn supports_formate_fn) {
    PADDLE_ENFORCE_NE(
        Contains(op_name),
        true,
        phi::errors::NotFound(
            "`%s` op's supportsFormatCombination has not been registered.",
            op_name));
    auto_meta_fn_map_.insert({std::move(op_name), supports_formate_fn});
  }

  const AutoGenerateSupportsFormateFn& GetAuto(
      const std::string& op_name) const {
    auto it = auto_meta_fn_map_.find(op_name);
    PADDLE_ENFORCE_NE(
        it,
        auto_meta_fn_map_.end(),
        phi::errors::NotFound(
            "`%s` op's supportsFormatCombination has not been registered.",
            op_name));
    return it->second;
  }

 private:
  SupportsFormateFnFactory() = default;

  paddle::flat_hash_map<std::string, SupportsFormateFn> meta_fn_map_;

  paddle::flat_hash_map<std::string, AutoGenerateSupportsFormateFn>
      auto_meta_fn_map_;

  DISABLE_COPY_AND_ASSIGN(SupportsFormateFnFactory);
};

using GetOutputDimsFn =
    nvinfer1::DimsExprs (*)(int32_t outputIndex,
                            const nvinfer1::DimsExprs* inputs,
                            int32_t nbInputs,
                            nvinfer1::IExprBuilder& exprBuilder);  // NOLINT

using AutoGenerateGetOutputDimsFn =
    nvinfer1::DimsExprs (*)(int32_t outputIndex,
                            const nvinfer1::DimsExprs* inputs,
                            int32_t nbInputs,
                            nvinfer1::IExprBuilder& exprBuilder,  // NOLINT
                            const paddle::OpMetaInfo& info,
                            const framework::OpDesc& desc);

class GetOutputDimsFnFactory {
 public:
  static GetOutputDimsFnFactory& Instance() {
    static GetOutputDimsFnFactory g_meta_fn_map;
    return g_meta_fn_map;
  }

  bool Contains(const std::string& op_name) const {
    return meta_fn_map_.count(op_name) > 0;
  }

  void Insert(std::string op_name, GetOutputDimsFn get_output_dims_fn) {
    PADDLE_ENFORCE_NE(
        Contains(op_name),
        true,
        phi::errors::AlreadyExists(
            "`%s` op's getOutputDimensions has been registered.", op_name));
    meta_fn_map_.insert({std::move(op_name), get_output_dims_fn});
  }

  const GetOutputDimsFn& Get(const std::string& op_name) const {
    auto it = meta_fn_map_.find(op_name);
    PADDLE_ENFORCE_NE(
        it,
        meta_fn_map_.end(),
        phi::errors::NotFound(
            "`%s` op's getOutputDimensions has not been registered.", op_name));
    return it->second;
  }

  bool ContainsAuto(const std::string& op_name) const {
    return auto_meta_fn_map_.count(op_name) > 0;
  }

  void InsertAuto(std::string op_name,
                  AutoGenerateGetOutputDimsFn get_output_dims_fn) {
    PADDLE_ENFORCE_NE(
        Contains(op_name),
        true,
        phi::errors::AlreadyExists(
            "`%s` op's getOutputDimensions has been registered.", op_name));
    auto_meta_fn_map_.insert({std::move(op_name), get_output_dims_fn});
  }

  const AutoGenerateGetOutputDimsFn& GetAuto(const std::string& op_name) const {
    auto it = auto_meta_fn_map_.find(op_name);
    PADDLE_ENFORCE_NE(
        it,
        auto_meta_fn_map_.end(),
        phi::errors::NotFound(
            "`%s` op's getOutputDimensions has not been registered.", op_name));
    return it->second;
  }

 private:
  GetOutputDimsFnFactory() = default;

  paddle::flat_hash_map<std::string, GetOutputDimsFn> meta_fn_map_;

  paddle::flat_hash_map<std::string, AutoGenerateGetOutputDimsFn>
      auto_meta_fn_map_;

  DISABLE_COPY_AND_ASSIGN(GetOutputDimsFnFactory);
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
