// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace gencode {

/// Zero Copy Tensor.
class Tensor {
 public:
  using ddim_t = std::vector<int64_t>;

  Tensor(const void *raw_tensor, void *raw_mutable_tensor)
      : raw_tensor_(raw_tensor), raw_mutable_tensor_(raw_mutable_tensor) {}

  void Resize(const ddim_t &shape);
  template <typename T>
  const T *data() const;
  template <typename T>
  T *mutable_data();

  ddim_t shape() const;

 private:
  const void *raw_tensor_;
  void *raw_mutable_tensor_{};
};

/*
 * Predictor for the generated code.
 */
class PaddlePredictor {
 public:
  void Init();

  std::unique_ptr<Tensor> GetTensor(const std::string &id) const;
  std::unique_ptr<Tensor> GetMutableTensor(const std::string &id);

  // Get offset-th col of feed.
  std::unique_ptr<Tensor> GetInput(size_t offset);

  std::unique_ptr<Tensor> GetOutput(size_t offset);

  void Run();

  PaddlePredictor();
  ~PaddlePredictor();

 private:
  void *raw_ops_;
  void *raw_kernels_;
  void *raw_scope_{};
  void *raw_exe_scope_{};  // raw_exe_scope is not owned.
};

}  // namespace gencode
}  // namespace paddle
