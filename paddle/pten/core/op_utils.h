/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <mutex>

#include "paddle/pten/core/arg_map_context.h"
#include "paddle/pten/core/kernel_def.h"
#include "paddle/pten/core/macros.h"
#include "paddle/utils/flat_hash_map.h"

namespace pten {

// TODO(chenweihang): Add function to remove suffix

class KernelSignatureMap {
 public:
  static KernelSignatureMap& Instance();

  bool Has(const std::string& op_type) const;

  const KernelSignature& Get(const std::string& op_type) const;

 private:
  KernelSignatureMap() = default;
  DISABLE_COPY_AND_ASSIGN(KernelSignatureMap);

 private:
  static KernelSignatureMap* kernel_signature_map_;
  static std::once_flag init_flag_;

  paddle::flat_hash_map<std::string, KernelSignature> map_;
};

// A set of components for compatibility with the original fluid op
class OpUtils {
 public:
  static OpUtils& Instance();

  bool Contains(const std::string& op_type) const;

  void InsertArgumentMappingFn(const std::string& op_type,
                               ArgumentMappingFn fn);
  void InsertInferMetaFn(const std::string& kernel_name_prefix, InferMetaFn fn);

  ArgumentMappingFn GetArgumentMappingFn(const std::string& op_type) const;
  InferMetaFn GetInferMetaFn(const std::string& kernel_name_prefix) const;

 private:
  OpUtils() = default;

  /**
   * [ Why kernel name prefix? ]
   *
   * one op -> a matrix of kernels
   *
   * such as, scale op, it may correspond to the following kernels:
   *
   * - scale, scale_sr, scale_dnnl
   * - scale_raw, scale_raw_sr, scale_raw_dnnl
   *
   * All the kernels in each row correspond to the same infershape function,
   * the number of kernel arguments in the same row is the same, and only
   * the tensor types in the arguments are different.
   */

  paddle::flat_hash_map<std::string, ArgumentMappingFn> args_fn_map_;
  paddle::flat_hash_map<std::string, InferMetaFn> infer_meta_fn_map_;

  DISABLE_COPY_AND_ASSIGN(OpUtils);
};

}  // namespace pten
