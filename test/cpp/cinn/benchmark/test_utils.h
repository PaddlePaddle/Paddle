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

#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace tests {

class OpBenchmarkTester {
 public:
  OpBenchmarkTester(const std::string &op_name,
                    const std::vector<std::vector<int>> &input_shapes,
                    const common::Target &target = common::DefaultHostTarget(),
                    int repeat = 10,
                    float diff = 1e-5)
      : op_name_(op_name),
        input_shapes_(input_shapes),
        target_(target),
        repeat_(repeat),
        diff_(diff) {}

  virtual ~OpBenchmarkTester() = default;

  void TestOp(const std::string &test_name,
              const std::vector<ir::Tensor> &input_tensors,
              const hlir::framework::NodeAttr &attrs,
              const std::vector<Type> &input_types,
              const std::vector<Type> &out_types,
              bool use_default_stragegy = true);

  virtual Module CreateCinnModule(const std::vector<ir::Tensor> &input_tensors,
                                  const hlir::framework::NodeAttr &attrs,
                                  const std::vector<Type> &out_types,
                                  bool use_default_stragegy = true);

  // should define specific stragey if not use default schedule
  virtual std::vector<ir::Tensor> CreateSpecificStrategy(
      const std::vector<ir::Tensor> &inputs, poly::StageMap *stages) {
    CINN_NOT_IMPLEMENTED
  }

  virtual std::unique_ptr<backends::ExecutionEngine> CreateExecutionEngine(
      const cinn::ir::Module &module);

  std::vector<cinn_pod_value_t> &GetAllArgs() { return all_args_; }
  int GetOutDims() { return out_dims_; }

  template <typename T = float>
  std::vector<ir::Tensor> CreateInputTensors() {
    std::vector<ir::Tensor> inputs;
    std::vector<std::vector<Expr>> expr_shapes;
    for (int i = 0; i < input_shapes_.size(); i++) {
      std::vector<Expr> expr_shape;
      for (int j = 0; j < input_shapes_[i].size(); ++j) {
        expr_shape.push_back(Expr(input_shapes_[i][j]));
      }
      expr_shapes.push_back(expr_shape);
      Placeholder<T> input(common::UniqName("input"), expr_shape);
      inputs.push_back(input.tensor());
    }
    return inputs;
  }

 private:
  void CreateBuffer();

  common::Target target_;
  std::string op_name_;
  float diff_;
  int repeat_;
  std::vector<std::vector<int>> input_shapes_;
  std::vector<std::vector<int>> output_shapes_;
  std::vector<Type> input_types_;
  std::vector<Type> out_types_;
  std::vector<cinn_pod_value_t> all_args_;
  int out_dims_;
};

}  // namespace tests
}  // namespace cinn
