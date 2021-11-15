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

#include <sstream>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/pten/api/lib/utils/allocator.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(EagerTensor, Constructor) {
  EagerTensor et1 = EagerTensor();
  EagerTensor et2 = EagerTensor("et2");
  pten::DenseTensorMeta meta = pten::DenseTensorMeta(
      pten::DataType::FLOAT32, paddle::framework::make_ddim({2, 2}));
  std::shared_ptr<pten::DenseTensor> dt = std::make_shared<pten::DenseTensor>(
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace()),
      meta);
  EagerTensor et3 = EagerTensor(dt);
}
