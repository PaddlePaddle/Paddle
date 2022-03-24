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

#include <ostream>
#include <string>
#include <tuple>

#include "paddle/phi/common/place.h"
#include "paddle/utils/any.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

namespace phi {

constexpr char kGradVarSuffix[] = "@GRAD";

constexpr size_t kGradVarSuffixSize = 5U;

inline std::string GradVarName(const std::string& var_name) {
  std::string result;
  result.reserve(var_name.size() + kGradVarSuffixSize);
  result += var_name;
  result += kGradVarSuffix;
  return result;
}

// tuple(input_names, attr_names, output_names)
using KernelArgsTuple = std::tuple<paddle::SmallVector<std::string>,
                                   paddle::SmallVector<std::string>,
                                   paddle::SmallVector<std::string>>;

struct KernelSignature {
  std::string name;
  KernelArgsTuple args;

  KernelSignature() = default;

  KernelSignature(std::string&& kernel_name,
                  paddle::SmallVector<std::string>&& inputs,
                  paddle::SmallVector<std::string>&& attrs,
                  paddle::SmallVector<std::string>&& outputs)
      : name(std::move(kernel_name)),
        args(std::make_tuple(inputs, attrs, outputs)) {}
  KernelSignature(const std::string& kernel_name,
                  const paddle::SmallVector<std::string>& inputs,
                  const paddle::SmallVector<std::string>& attrs,
                  const paddle::SmallVector<std::string>& outputs)
      : name(kernel_name), args(std::make_tuple(inputs, attrs, outputs)) {}

  // TODO(chenweihang): add assign constructor to solve windows compile
  // problem, remove it later
  KernelSignature& operator=(const KernelSignature& other) {
    name = other.name;
    args = other.args;
    return *this;
  }
};

std::ostream& operator<<(std::ostream& os, KernelSignature signature);

// TODO(chenweihang): Add more methods if needed in future
class ArgumentMappingContext {
 public:
  virtual ~ArgumentMappingContext() = default;

  virtual bool HasInput(const std::string& name) const = 0;
  virtual bool HasOutput(const std::string& name) const = 0;
  virtual bool HasAttr(const std::string& name) const = 0;

  // now we can't use Attribute here, it will cause phi relay on
  // boost::variant and BlockDesc
  virtual paddle::any Attr(const std::string& name) const = 0;

  virtual size_t InputSize(const std::string& name) const = 0;
  virtual size_t OutputSize(const std::string& name) const = 0;

  virtual bool IsDenseTensorInput(const std::string& name) const = 0;
  virtual bool IsSelectedRowsInput(const std::string& name) const = 0;
  // For compatibility with LoDTensorArray
  virtual bool IsDenseTensorVectorInput(const std::string& name) const = 0;

  virtual bool IsDenseTensorOutput(const std::string& name) const = 0;
  virtual bool IsSelectedRowsOutput(const std::string& name) const = 0;

  // use this function to mark it comes from InferShapeArgumentMappingContext
  // and will be used in infershape
  virtual bool IsForInferShape() const = 0;

  // NOTE(paddle-dev): [ Why do we export this interface? ]
  // In old Fluid framework, some operators' Attribute can be a Tensor or
  // TensorList. In this case, the InferShape logic will be different
  // under CompileTime and RuntimeTime. So we export this interface to
  // handle it conveniently. See "gaussian_random_sig.cc" for details.
  virtual bool IsRuntime() const { return true; }
};

}  // namespace phi
