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
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/macros.h"
#include "paddle/utils/flat_hash_map.h"

namespace paddle {
namespace inference {
namespace tensorrt {

using DynamicMetaFn =
    nvinfer1::DimsExprs (*)(int output_index,
                            const nvinfer1::DimsExprs* inputs,
                            int nb_inputs,
                            nvinfer1::IExprBuilder& expr_builder,  // NOLINT
                            const framework::OpDesc& op_desc);

class DynamicMetaFnFactory {
 public:
  static DynamicMetaFnFactory& Instance() {
    static DynamicMetaFnFactory g_meta_fn_map;
    return g_meta_fn_map;
  }

  bool Contains(const std::string& op_name) const {
    return meta_fn_map_.count(op_name) > 0;
  }

  void Insert(std::string op_name, DynamicMetaFn infer_meta_fn) {
    PADDLE_ENFORCE_NE(
        Contains(op_name),
        true,
        phi::errors::AlreadyExists(
            "`%s` op's DynamicInferMetaFn has been registered.", op_name));
    meta_fn_map_.insert({std::move(op_name), std::move(infer_meta_fn)});
  }

  const DynamicMetaFn& Get(const std::string& op_name) const {
    auto it = meta_fn_map_.find(op_name);
    PADDLE_ENFORCE_NE(
        it,
        meta_fn_map_.end(),
        phi::errors::NotFound(
            "`%s` op's DynamicInferMetaFn has been registered.", op_name));
    return it->second;
  }

 private:
  DynamicMetaFnFactory() = default;

  paddle::flat_hash_map<std::string, DynamicMetaFn> meta_fn_map_;

  DISABLE_COPY_AND_ASSIGN(DynamicMetaFnFactory);
};

struct DynamicMetaFnRegistrar {
  DynamicMetaFnRegistrar(const char* op_name, DynamicMetaFn infer_meta_fn) {
    DynamicMetaFnFactory::Instance().Insert(op_name, std::move(infer_meta_fn));
  }

  static void Touch() {}
};

#define PD_REGISTER_DYNAMIC_INFER_META_FN(op_name, dynamic_infer_meta_fn)   \
  static paddle::inference::tensorrt::DynamicMetaFnRegistrar                \
      registrar_dynamic_infer_meta_fn_for_##op_name(#op_name,               \
                                                    dynamic_infer_meta_fn); \
  int TouchDynamicMetaFnRegistrar_##op_name() {                             \
    registrar_dynamic_infer_meta_fn_for_##op_name.Touch();                  \
    return 0;                                                               \
  }

#define USE_TRT_DYNAMIC_INFER_META_FN(op_name)           \
  extern int TouchDynamicMetaFnRegistrar_##op_name();    \
  static int use_op_dynamic_infer_meta##op_name UNUSED = \
      TouchDynamicMetaFnRegistrar_##op_name();

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
