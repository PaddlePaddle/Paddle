// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/decomposer/test_helper.h"

namespace cinn::frontend {

void RunDecomposer(Program* prog,
                   const Target& target,
                   const std::vector<std::string>& passes,
                   const std::vector<std::string>& fetch_ids) {
  VLOG(1) << "===================== Before Program Pass =====================";
  for (int i = 0; i < prog->size(); i++) {
    VLOG(1) << "instruction: " << (*prog)[i];
  }
  ProgramPass::Apply(
      prog,
      std::unordered_set<std::string>(fetch_ids.begin(), fetch_ids.end()),
      target,
      passes);
  VLOG(1) << "===================== After Program Pass =====================";
  for (int i = 0; i < prog->size(); i++) {
    VLOG(1) << "instruction: " << (*prog)[i];
  }
}

template <>
void InitRandomVector<int>(
    std::vector<int>* vec, size_t numel, int low, int high, float precision) {
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_int_distribution<int> dist(low, high);

  vec->resize(numel);
  for (size_t i = 0; i < numel; ++i) {
    vec->at(i) = dist(engine);
  }
}

template <>
void CopyFromVector<bool>(const std::vector<bool>& vec,
                          hlir::framework::Tensor tensor,
                          Target target) {
  auto* data = tensor->mutable_data<bool>(target);

  size_t numel = tensor->shape().numel();
  CHECK_EQ(vec.size(), numel);

#ifdef CINN_WITH_CUDA
  // why not use vector<bool> ? Because to optimizes space, each value is stored
  // in a single bit. So that the vector<bool> doesn't has data() function.
  CHECK_EQ(sizeof(bool), sizeof(char))
      << "The test need ensure the byte size of bool equal to the byte size of "
         "char.";

  std::vector<char> vec_char(numel);
  for (int i = 0; i < numel; ++i) vec_char[i] = static_cast<char>(vec[i]);
  cudaMemcpy(
      data, vec_char.data(), numel * sizeof(bool), cudaMemcpyHostToDevice);
#else
  std::copy(vec.begin(), vec.end(), data);
#endif
}

template <>
void CopyToVector<bool>(const hlir::framework::Tensor tensor,
                        std::vector<bool>* vec) {
  auto* data = tensor->data<bool>();

  size_t numel = tensor->shape().numel();
  vec->resize(numel);

#ifdef CINN_WITH_CUDA
  // why not use vector<bool> ? Because to optimizes space, each value is stored
  // in a single bit. So that the vector<bool> doesn't has data() function.
  CHECK_EQ(sizeof(bool), sizeof(char))
      << "The test need ensure the byte size of bool equal to the byte size of "
         "char.";

  std::vector<char> vec_char(numel);
  cudaMemcpy(
      vec_char.data(), data, numel * sizeof(bool), cudaMemcpyDeviceToHost);
  for (int i = 0; i < numel; ++i) vec->at(i) = static_cast<bool>(vec_char[i]);
#else
  for (size_t i = 0; i < numel; ++i) {
    vec->at(i) = data[i];
  }
#endif
}

}  // namespace cinn::frontend
