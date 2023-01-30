// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <unordered_map>

#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace phi {

class DenseTensorMap {
 public:
  DenseTensorMap() = default;
  DenseTensorMap(DenseTensorMap&& other) : map_(std::move(other.map_)) {}
  void SetDenseTensor(const std::string& name,
<<<<<<< HEAD
                      std::unique_ptr<::Tensor>&& tensor);
  ::Tensor* GetDenseTensor(const std::string& name) const;
=======
                      std::unique_ptr<::phi::DenseTensor>&& tensor);
  ::phi::DenseTensor* GetDenseTensor(const std::string& name) const;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  size_t size() const;

 private:
  mutable std::mutex mu_;
<<<<<<< HEAD
  std::unordered_map<std::string, std::unique_ptr<::Tensor>> map_;
=======
  std::unordered_map<std::string, std::unique_ptr<::phi::DenseTensor>> map_;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
};

}  // namespace phi
}  // namespace infrt
