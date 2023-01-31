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

#include <atomic>
#include <memory>
#include <vector>

#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace operators {
namespace reader {

class LoDTensorBlockingQueue;

class PyReader : public framework::FileReader {
 public:
  explicit PyReader(
      const std::shared_ptr<LoDTensorBlockingQueue>& queue,
      const std::vector<framework::DDim>& dims,
      const std::vector<framework::proto::VarType::Type>& var_types,
      const std::vector<bool>& need_check_feed);

  void ReadNext(paddle::framework::LoDTensorArray* out) override;

  ~PyReader();

  void Shutdown() override;

  void Start() override;

 private:
  std::shared_ptr<LoDTensorBlockingQueue> queue_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle
