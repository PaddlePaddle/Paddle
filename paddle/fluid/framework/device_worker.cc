/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/device_worker.h"

namespace paddle {
namespace framework {

void DeviceWorker::SetRootScope(Scope* root_scope) { root_scope_ = root_scope; }

void DeviceWorker::SetDataFeed(DataFeed* data_feed) {
  device_reader_ = data_feed;
}

template <typename T>
std::string PrintLodTensorType(LoDTensor* tensor, int64_t start, int64_t end) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    return "access violation";
  }
  std::ostringstream os;
  for (int64_t i = start; i < end; i++) {
    os << ":" << tensor->data<T>()[i];
  }
  return os.str();
}

std::string PrintLodTensorIntType(LoDTensor* tensor, int64_t start,
                                  int64_t end) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    return "access violation";
  }
  std::ostringstream os;
  for (int64_t i = start; i < end; i++) {
    os << ":" << static_cast<uint64_t>(tensor->data<int64_t>()[i]);
  }
  return os.str();
}

std::string PrintLodTensor(LoDTensor* tensor, int64_t start, int64_t end) {
  std::string out_val;
  if (tensor->type() == proto::VarType::FP32) {
    out_val = PrintLodTensorType<float>(tensor, start, end);
  } else if (tensor->type() == proto::VarType::INT64) {
    out_val = PrintLodTensorIntType(tensor, start, end);
  } else if (tensor->type() == proto::VarType::FP64) {
    out_val = PrintLodTensorType<double>(tensor, start, end);
  } else {
    out_val = "unsupported type";
  }
  return out_val;
}

std::pair<int64_t, int64_t> GetTensorBound(LoDTensor* tensor, int index) {
  auto& dims = tensor->dims();
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    return {lod[index] * dims[1], lod[index + 1] * dims[1]};
  } else {
    return {index * dims[1], (index + 1) * dims[1]};
  }
}

bool CheckValidOutput(LoDTensor* tensor, size_t batch_size) {
  auto& dims = tensor->dims();
  if (dims.size() != 2) return false;
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    if (lod.size() != batch_size + 1) {
      return false;
    }
  } else {
    if (dims[0] != static_cast<int>(batch_size)) {
      return false;
    }
  }
  return true;
}

}  // namespace framework
}  // namespace paddle
