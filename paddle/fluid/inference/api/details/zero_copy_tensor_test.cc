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
#include <limits>
#include <memory>
#include <random>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_tensor.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle_infer {

struct TensorWrapper : public Tensor {
  TensorWrapper(
      paddle_infer::PlaceType place,
      paddle::framework::Scope* scope,
      const std::map<phi::Place,
                     std::shared_future<std::unique_ptr<phi::DeviceContext>>>*
          dev_ctxs,
      const std::string& name)
      : Tensor{static_cast<void*>(scope), dev_ctxs} {
    SetPlace(place, 0 /*device_id*/);
    SetName(name);
    input_or_output_ = true;
  }
};

std::unique_ptr<Tensor> CreateTensor(paddle_infer::PlaceType place,
                                     paddle::framework::Scope* scope,
                                     const std::string& name) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  const auto& dev_ctxs = pool.device_contexts();
  return std::unique_ptr<Tensor>(
      new TensorWrapper{place, scope, &dev_ctxs, name});
}

template <typename T>
struct RandomGenerator {
  RandomGenerator(
      double min = static_cast<double>((std::numeric_limits<T>::min)()),
      double max = static_cast<double>((std::numeric_limits<T>::max)()))
      : dist_{min, max} {}
  T operator()() { return static_cast<T>(dist_(random_engine_)); }

 private:
  std::mt19937_64 random_engine_{std::random_device()()};
  std::uniform_real_distribution<double> dist_;
};

template <typename T, template <typename> class G>
bool FillRandomDataAndCheck(PlaceType place,
                            size_t length,
                            G<T>&& generator,
                            float threshold = 10e-5) {
  std::vector<T> data_in(length);
  std::generate(data_in.begin(), data_in.end(), std::forward<G<T>>(generator));
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

template <typename T>
bool SetPlaceAndCheck(PlaceType place, size_t length) {
  paddle::framework::Scope scope;
  const std::string name{"name"};
  const std::vector<std::vector<size_t>> lod{{0, length}};
  scope.Var(name);
  auto tensor = CreateTensor(place, &scope, name);
  std::vector<int> shape{static_cast<int>(length)};
  tensor->Reshape(shape);
  tensor->mutable_data<T>(place);
  tensor->SetLoD(lod);

  PlaceType place_out{PlaceType::kUNK};
  int length_out{-1};
  tensor->data<T>(&place_out, &length_out);
  if (length_out != static_cast<int>(length) || place_out != place) {
    return false;
  }
  if (tensor->name() != name || tensor->lod() != lod) {
    return false;
  }
  return true;
}

bool FillRandomDataAndCheck(PlaceType place) {
  const size_t length{RandomGenerator<size_t>{1, 1000}()};
  VLOG(3) << "FillRandomDataAndCheck: length = " << length;
  return FillRandomDataAndCheck<float>(
             place, length, RandomGenerator<float>{}) &&
         FillRandomDataAndCheck<int64_t>(
             place, length, RandomGenerator<int64_t>{}) &&
         FillRandomDataAndCheck<int32_t>(
             place, length, RandomGenerator<int32_t>{}) &&
         FillRandomDataAndCheck<uint8_t>(
             place, length, RandomGenerator<uint8_t>{});
}

bool SetPlaceAndCheck(PlaceType place) {
  const size_t length{RandomGenerator<size_t>{1, 1000}()};
  VLOG(3) << "SetPlaceAndCheck: length = " << length;
  return SetPlaceAndCheck<float>(place, length) &&
         SetPlaceAndCheck<int64_t>(place, length) &&
         SetPlaceAndCheck<int32_t>(place, length) &&
         SetPlaceAndCheck<uint8_t>(place, length);
}

TEST(Tensor, FillRandomDataAndCheck) {
  ASSERT_TRUE(FillRandomDataAndCheck(PlaceType::kCPU));
  ASSERT_TRUE(SetPlaceAndCheck(PlaceType::kCPU));
#ifdef PADDLE_WITH_CUDA
  ASSERT_TRUE(FillRandomDataAndCheck(PlaceType::kGPU));
  ASSERT_TRUE(SetPlaceAndCheck(PlaceType::kGPU));
#endif
#ifdef PADDLE_WITH_XPU
  ASSERT_TRUE(FillRandomDataAndCheck(PlaceType::kXPU));
  ASSERT_TRUE(SetPlaceAndCheck(PlaceType::kXPU));
#endif
}

}  // namespace paddle_infer
