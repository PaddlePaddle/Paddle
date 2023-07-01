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

#include <string>
#include <vector>

#include "paddle/cinn/hlir/pe/transform.h"
#include "test/cpp/cinn/benchmark/test_utils.h"

namespace cinn {
namespace tests {

class MatmulTester : public OpBenchmarkTester {
 public:
  MatmulTester(const std::string &op_name,
               const std::vector<std::vector<int>> &input_shapes,
               const common::Target &target = common::DefaultHostTarget(),
               int repeat = 10,
               float diff = 1e-5)
      : OpBenchmarkTester(op_name, input_shapes, target, repeat, diff) {}

  std::vector<ir::Tensor> CreateSpecificStrategy(
      const std::vector<ir::Tensor> &inputs, poly::StageMap *stages) override;
};

class MatmulTileTester : public MatmulTester {
 public:
  MatmulTileTester(const std::string &op_name,
                   const std::vector<std::vector<int>> &input_shapes,
                   const common::Target &target = common::DefaultHostTarget(),
                   int repeat = 10,
                   float diff = 1e-5)
      : MatmulTester(op_name, input_shapes, target, repeat, diff) {}

  std::vector<ir::Tensor> CreateSpecificStrategy(
      const std::vector<ir::Tensor> &inputs, poly::StageMap *stages) override;
};

class MatmulSplitTester : public MatmulTester {
 public:
  MatmulSplitTester(const std::string &op_name,
                    const std::vector<std::vector<int>> &input_shapes,
                    const common::Target &target = common::DefaultHostTarget(),
                    int repeat = 10,
                    float diff = 1e-5)
      : MatmulTester(op_name, input_shapes, target, repeat, diff) {}

  virtual std::vector<ir::Tensor> CreateSpecificStrategy(
      const std::vector<ir::Tensor> &inputs, poly::StageMap *stages);
};

class MatmulBlockTester : public MatmulTester {
 public:
  MatmulBlockTester(const std::string &op_name,
                    const std::vector<std::vector<int>> &input_shapes,
                    const common::Target &target = common::DefaultHostTarget(),
                    int repeat = 10,
                    float diff = 1e-5)
      : MatmulTester(op_name, input_shapes, target, repeat, diff),
        input_shapes_(input_shapes) {}

  std::vector<ir::Tensor> CreateSpecificStrategy(
      const std::vector<ir::Tensor> &inputs, poly::StageMap *stages) override;

  std::vector<std::vector<int>> input_shapes_;
};

class MatmulVectorizeTester : public MatmulTester {
 public:
  MatmulVectorizeTester(
      const std::string &op_name,
      const std::vector<std::vector<int>> &input_shapes,
      const common::Target &target = common::DefaultHostTarget(),
      int repeat = 10,
      float diff = 1e-5)
      : MatmulTester(op_name, input_shapes, target, repeat, diff),
        input_shapes_(input_shapes) {}

  std::vector<ir::Tensor> CreateSpecificStrategy(
      const std::vector<ir::Tensor> &inputs, poly::StageMap *stages) override;
  std::vector<std::vector<int>> input_shapes_;
};

class MatmulLoopPermutationTester : public MatmulTester {
 public:
  MatmulLoopPermutationTester(
      const std::string &op_name,
      const std::vector<std::vector<int>> &input_shapes,
      const common::Target &target = common::DefaultHostTarget(),
      int repeat = 10,
      float diff = 1e-5)
      : MatmulTester(op_name, input_shapes, target, repeat, diff),
        input_shapes_(input_shapes) {}

  std::vector<ir::Tensor> CreateSpecificStrategy(
      const std::vector<ir::Tensor> &inputs, poly::StageMap *stages) override;
  std::vector<std::vector<int>> input_shapes_;
};

class MatmulArrayPackingTester : public MatmulTester {
 public:
  MatmulArrayPackingTester(
      const std::string &op_name,
      const std::vector<std::vector<int>> &input_shapes,
      const common::Target &target = common::DefaultHostTarget(),
      int repeat = 10,
      float diff = 1e-5)
      : MatmulTester(op_name, input_shapes, target, repeat, diff),
        input_shapes_(input_shapes) {}

  std::vector<ir::Tensor> CreateSpecificStrategy(
      const std::vector<ir::Tensor> &inputs, poly::StageMap *stages) override;
  std::vector<std::vector<int>> input_shapes_;
};

}  // namespace tests
}  // namespace cinn
