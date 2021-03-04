// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle_infer {

struct TensorWrapper : public Tensor {
  TensorWrapper(paddle_infer::PlaceType place, paddle::framework::Scope* scope,
                const std::string& name)
      : Tensor{static_cast<void*>(scope)} {
    SetPlace(place, 0 /*device_id*/);
    SetName(name);
  }
};

std::unique_ptr<Tensor> CreateTensor(paddle_infer::PlaceType place,
                                     paddle::framework::Scope* scope,
                                     const std::string& name) {
  return std::unique_ptr<Tensor>(new TensorWrapper{place, scope, name});
}

template <typename T>
struct UniformRealGenerator {
  std::function<T()> operator()() const {
    static std::uniform_real_distribution<T> distribution(
        std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    static std::default_random_engine generator;
    return []() { return distribution(generator); };
  }
};

template <typename T>
struct UniformIntGenerator {
  std::function<T()> operator()() const {
    static std::uniform_int_distribution<T> distribution(
        std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    static std::default_random_engine generator;
    return []() { return distribution(generator); };
  }
};

template <typename T, template <typename> typename G>
bool FillRandomDataAndCheck(paddle_infer::PlaceType place, size_t length,
                            const G<T>& generator, float threshold = 10e-5) {
  std::vector<T> data_in(length);
  std::generate(data_in.begin(), data_in.end(), generator());
  paddle::framework::Scope scope;
  const std::string name{"name"};
  scope.Var(name);
  auto tensor = CreateTensor(place, &scope, name);
  tensor->CopyFromCpu<T>(data_in.data());
  if (tensor->type() != paddle::inference::ConvertToPaddleDType(
                            paddle::framework::DataTypeTrait<T>::DataType())) {
    return false;
  }
  std::vector<T> data_out(length);
  tensor->CopyToCpu<T>(data_out.data());
  for (size_t i = 0; i < length; ++i) {
    if (std::abs(data_out[i] - data_out[i]) > threshold) {
      return false;
    }
  }
  return true;
}

bool FillRandomDataAndCheck(PlaceType place) {
  return FillRandomDataAndCheck<float>(place, 100,
                                       UniformRealGenerator<float>()) &&
         FillRandomDataAndCheck<int64_t>(place, 100,
                                         UniformIntGenerator<int64_t>()) &&
         FillRandomDataAndCheck<int32_t>(place, 100,
                                         UniformIntGenerator<int32_t>()) &&
         FillRandomDataAndCheck<uint8_t>(place, 100,
                                         UniformIntGenerator<uint8_t>());
}

TEST(Tensor, FillRandomDataAndCheck) {
  ASSERT_TRUE(FillRandomDataAndCheck(PlaceType::kCPU));
#ifdef PADDLE_WITH_CUDA
  ASSERT_TRUE(FillRandomDataAndCheck(PlaceType::kGPU));
#endif
}

}  // namespace paddle_infer
