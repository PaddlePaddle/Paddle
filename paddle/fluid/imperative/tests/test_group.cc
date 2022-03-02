// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include "gtest/gtest.h"

#include "paddle/fluid/imperative/reducer.h"

namespace paddle {
namespace imperative {

TEST(TestGroup, TestPrintGroupMessage) {
  Group group;
  std::stringstream stream1, stream2;
  stream1 << group;
  ASSERT_STREQ(stream1.str().c_str(),
               "numel: 0 ;is_sparse: 0 ;var number: 0\n[]\n");

  std::vector<size_t> vars;
  size_t vars_num = 102;
  for (size_t i = 0; i < vars_num; ++i) {
    vars.push_back(i);
  }
  group.variable_indices_ = vars;
  group.all_length_ = 102;
  group.is_sparse_ = false;

  std::string head = "numel: 102 ;is_sparse: 0 ;var number: 102\n";
  head = head + "[";
  auto begin = vars.begin();
  auto end = vars.end();
  for (int i = 0; begin != end && i < 100; ++i, ++begin) {
    if (i > 0) head += ' ';
    head += std::to_string(*begin);
  }
  if (begin != end) {
    head += " ...";
  }
  head += "]\n";
  stream2 << group;
  ASSERT_STREQ(stream2.str().c_str(), head.c_str());
}

template <typename T, typename Place>
void GroupConcatSplit(Place place, size_t size) {
  platform::CPUPlace cpu_place;
  Group group;

  // [[0.0], [0.0, 1.0], [0.0, 1.0, 2.0] .. ]
  std::vector<framework::Variable> vars;
  vars.resize(size);
  for (size_t i = 0; i < size; ++i) {
    auto len = i + 1;
    auto* tensor = vars[i].GetMutable<framework::LoDTensor>();
    tensor->Resize({static_cast<int64_t>(len)});
    auto* data = tensor->mutable_data<T>(place);

    std::vector<T> value;
    for (size_t j = 0; j < len; ++j) {
      value.push_back(static_cast<T>(1.0 * j));
    }

    if (std::is_same<Place, platform::CUDAPlace>::value ||
        std::is_same<Place, platform::MLUPlace>::value) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_CNCL)
      paddle::memory::Copy(place, data, cpu_place, value.data(),
                           sizeof(T) * value.size(), 0);
#endif
    } else {
      paddle::memory::Copy(place, data, cpu_place, value.data(),
                           sizeof(T) * value.size());
    }

    framework::Tensor tmp;
    tmp.ShareDataWith(*tensor).Resize({static_cast<int64_t>(len)});
    group.dense_tensors_.push_back(std::move(tmp));
    group.all_length_ += len;
    group.dtype_ = framework::TransToProtoVarType(tensor->dtype());
  }

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  {  // concat
    auto* tensor = group.dense_contents_.GetMutable<framework::LoDTensor>();
    tensor->Resize(phi::make_ddim({group.all_length_}))
        .mutable_data(place, framework::TransToPhiDataType(group.dtype_));
    group.ConcatTensors(*dev_ctx);

    group.DivNRanks(*dev_ctx, 1);

    framework::Tensor tmp;
    framework::TensorCopySync(*tensor, cpu_place, &tmp);
    auto* data = tmp.data<T>();
    size_t offset = 0;
    for (size_t i = 0; i < size; ++i) {
      auto len = i + 1;
      for (size_t j = 0; j < len; ++j) {
        EXPECT_EQ(data[offset + j], static_cast<T>(1.0 * j));
        // [[-0.0], [-0.0, -1.0], [-0.0, -1.0, -2.0] .. ]
        data[offset + j] = -data[offset + j];
      }
      offset += len;
    }
    framework::TensorCopySync(tmp, place, tensor);
  }

  {  // split
    group.SplitTensors(*dev_ctx);
    for (size_t i = 0; i < size; ++i) {
      auto len = i + 1;
      auto& tensor = group.dense_tensors_[i];
      framework::Tensor tmp;
      framework::TensorCopySync(tensor, cpu_place, &tmp);
      auto* data = tmp.data<T>();

      for (size_t j = 0; j < len; ++j) {
        EXPECT_EQ(data[j], static_cast<T>(-1.0 * j));
      }
    }
  }
}

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
TEST(TestGroup, TestConcatSplit) {
  platform::CUDAPlace cuda_place(0);
  platform::CPUPlace cpu_place;

  int size = 3;
  GroupConcatSplit<float>(cpu_place, size);
  GroupConcatSplit<double>(cpu_place, size);
  GroupConcatSplit<platform::float16>(cpu_place, size);

  GroupConcatSplit<float>(cuda_place, size);
  GroupConcatSplit<double>(cuda_place, size);
  GroupConcatSplit<platform::float16>(cuda_place, size);

  size = 15;
  GroupConcatSplit<float>(cpu_place, size);
  GroupConcatSplit<double>(cpu_place, size);
  GroupConcatSplit<platform::float16>(cpu_place, size);

  GroupConcatSplit<float>(cuda_place, size);
  GroupConcatSplit<double>(cuda_place, size);
  GroupConcatSplit<platform::float16>(cuda_place, size);
}

TEST(TestGroup, TestConcatSplitException) {
  platform::CUDAPinnedPlace place;

  int size = 3;
  ASSERT_ANY_THROW(GroupConcatSplit<float>(place, size));
}
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
TEST(TestGroup, TestXPUConcatSplit) {
  platform::XPUPlace xpu_place(0);
  platform::CPUPlace cpu_place;

  int size = 3;
  GroupConcatSplit<float>(cpu_place, size);
  GroupConcatSplit<float>(xpu_place, size);

  size = 15;
  GroupConcatSplit<float>(cpu_place, size);
  GroupConcatSplit<float>(xpu_place, size);
}
#endif

#if defined(PADDLE_WITH_CNCL)
TEST(TestGroup, TestMLUConcatSplit) {
  platform::MLUPlace mlu_place(0);
  platform::CPUPlace cpu_place;

  int size = 3;
  GroupConcatSplit<float>(cpu_place, size);
  GroupConcatSplit<float>(mlu_place, size);

  size = 15;
  GroupConcatSplit<float>(cpu_place, size);
  GroupConcatSplit<float>(mlu_place, size);
}
#endif
}  // namespace imperative
}  // namespace paddle
