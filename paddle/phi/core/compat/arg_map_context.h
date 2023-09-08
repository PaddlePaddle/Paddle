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
#include "paddle/phi/core/type_defs.h"
#include "paddle/utils/any.h"
#include "paddle/utils/flat_hash_map.h"
#include "paddle/utils/small_vector.h"

namespace phi {

// tuple(input_names, attr_names, output_names)
using KernelArgsTuple = std::tuple<paddle::small_vector<const char*>,
                                   paddle::small_vector<const char*>,
                                   paddle::small_vector<const char*>>;

struct KernelSignature {
  const char* name;
  paddle::small_vector<const char*> input_names;
  paddle::small_vector<const char*> attr_names;
  paddle::small_vector<const char*> output_names;

  KernelSignature() = default;

  KernelSignature(const char* kernel_name,
                  paddle::small_vector<const char*>&& inputs,
                  paddle::small_vector<const char*>&& attrs,
                  paddle::small_vector<const char*>&& outputs)
      : name(kernel_name),
        input_names(std::move(inputs)),
        attr_names(std::move(attrs)),
        output_names(std::move(outputs)) {}
  KernelSignature(const char* kernel_name,
                  const paddle::small_vector<const char*>& inputs,
                  const paddle::small_vector<const char*>& attrs,
                  const paddle::small_vector<const char*>& outputs)
      : name(kernel_name),
        input_names(inputs),
        attr_names(attrs),
        output_names(outputs) {}

  explicit KernelSignature(const char* kernel_name) : name(kernel_name) {}

  // TODO(chenweihang): add assign constructor to solve windows compile
  // problem, remove it later
  KernelSignature(const KernelSignature& other)
      : name(other.name),
        input_names(other.input_names),
        attr_names(other.attr_names),
        output_names(other.output_names) {}

  KernelSignature(KernelSignature&& other) noexcept
      : name(other.name),
        input_names(std::move(other.input_names)),
        attr_names(std::move(other.attr_names)),
        output_names(std::move(other.output_names)) {}

  KernelSignature& operator=(const KernelSignature& other) {
    name = other.name;
    input_names = other.input_names;
    attr_names = other.attr_names;
    output_names = other.output_names;
    return *this;
  }

  KernelSignature& operator=(KernelSignature&& other) noexcept {
    name = other.name;
    input_names = std::move(other.input_names);
    attr_names = std::move(other.attr_names);
    output_names = std::move(other.output_names);
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
  // paddle::variant and BlockDesc
  virtual paddle::any Attr(const std::string& name) const = 0;

  virtual size_t InputSize(const std::string& name) const = 0;
  virtual size_t OutputSize(const std::string& name) const = 0;

  virtual bool IsDenseTensorInput(const std::string& name) const = 0;
  virtual bool IsDenseTensorInputs(const std::string& name) const = 0;
  virtual bool IsSelectedRowsInput(const std::string& name) const = 0;
  virtual bool IsSelectedRowsInputs(const std::string& name) const = 0;
  virtual bool IsSparseCooTensorInput(const std::string& name) const = 0;
  virtual bool IsSparseCooTensorOutput(const std::string& name) const = 0;
  virtual bool IsSparseCsrTensorInput(const std::string& name) const = 0;
  // For compatibility with LoDTensorArray
  virtual bool IsDenseTensorVectorInput(const std::string& name) const = 0;
  virtual bool IsDenseTensorVectorOutput(const std::string& name) const = 0;

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
