// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <absl/container/flat_hash_map.h>
#include <absl/strings/string_view.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace backends {

struct FunctionProto {
  using shape_inference_t = std::function<std::vector<Expr> /*shape*/ (
      const std::vector<Expr>& /*arguments*/, int /*value_offset*/)>;

  std::string name;
  std::vector<Type> readonly_arg_types;
  std::vector<Type> mutable_arg_types;
  Type ret_type;

  // Inference the output's shape.
  shape_inference_t shape_inference;

  /**
   * Constructor for multiple output function.
   * @param name Name of the function.
   * @param readonly_arg_types The input types.
   * @param mutable_arg_types The output types.
   * @param ret_type The return type, default to Void().
   * @param shape_inference The shape inference for each of the output tensor.
   */
  FunctionProto(const std::string& name,
                const std::vector<Type>& readonly_arg_types,
                const std::vector<Type>& mutable_arg_types,
                Type ret_type = Void(),
                shape_inference_t shape_inference = shape_inference_t());

  /**
   * Constructor for single output function.
   * @param name Name of the function.
   * @param input_types The input types.
   * @param ret_type The return type.
   */
  FunctionProto(const std::string& name,
                const std::vector<Type>& input_types,
                Type ret_type)
      : name(name), readonly_arg_types(input_types), ret_type(ret_type) {}

  /**
   * Tell whether the Call \p op matches the function prototype.
   */
  bool Match(const ir::Call* op) const;

  /**
   * Assert the call should match the function prototype.
   */
  void AssertMatch(const ir::Call* op) const;

  struct Builder {
    explicit Builder(const std::string& name) {
      data_.reset(new FunctionProto);
      data_->name = name;
    }
    template <typename T>
    Builder& SetRetType() {
      data_->ret_type = type_of<T>();
      return *this;
    }
    template <typename T>
    Builder& AddInputType() {
      data_->readonly_arg_types.push_back(type_of<T>());
      return *this;
    }
    template <typename T>
    Builder& AddOutputType() {
      data_->mutable_arg_types.push_back(type_of<T>());
      return *this;
    }
    Builder& SetShapeInference(shape_inference_t fn) {
      data_->shape_inference = fn;
      return *this;
    }

    std::unique_ptr<FunctionProto> Build() { return std::move(data_); }

   private:
    std::unique_ptr<FunctionProto> data_;
  };

  /**
   * All the outputs use the n-th argument's shape.
   */
  static shape_inference_t ShapeFollowNthArgument(int n);

 protected:
  void CheckValid();

  FunctionProto() = default;
};

class FunctionProtoRegistry {
 public:
  FunctionProto* Register(absl::string_view name, FunctionProto* x);

  FunctionProto* Lookup(const std::string& name);

  std::string debug_string() const;

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<FunctionProto>> data_;
};

}  // namespace backends
}  // namespace cinn
