/* Copyright (c) 2023 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/gcu/runtime/rt_utils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {
namespace {
struct GcuCallback {
  std::function<void()> callback;

  explicit GcuCallback(const std::function<void()> &callback)
      : callback(std::move(callback)) {}
};

void RunGcuCallback(topsStream_t stream,
                    topsError_t error,
                    void *callback_ptr) {
  std::unique_ptr<GcuCallback> gcu_callback(
      reinterpret_cast<GcuCallback *>(callback_ptr));
  gcu_callback->callback();
  gcu_callback.reset();
}
}  // namespace

void AddGcuCallback(int device,
                    topsStream_t stream,
                    const std::function<void()> &callback) {
  auto gcu_callback = std::make_unique<GcuCallback>(std::move(callback));
  GcuDeviceGuard guard(device);
  RT_CHECK(
      topsStreamAddCallback(stream, RunGcuCallback, gcu_callback.release(), 0));
}

std::string DumpHbm(void *ptr) {
  std::stringstream ss;
  if (ptr) {
    size_t ndims = 16;
    std::vector<int64_t> dims(ndims, 0);
    ss << "  ptr address: " << ptr << "\n";
    RT_CHECK(topsMemoryGetDims(
        ptr, reinterpret_cast<int64_t *>(dims.data()), &ndims));
    ss << "  scatter dims: "
       << VectorToString(
              std::vector<int64_t>(dims.begin(), dims.begin() + ndims))
       << "\n";
    ss << "    {\n";
    uint64_t sub_num = 0;
    RT_CHECK(topsScatterMemoryGetSubNum(ptr, &sub_num));
    for (uint64_t idx = 0; idx < sub_num; ++idx) {
      void *sub_mem;
      RT_CHECK(topsScatterGetSubMem(ptr, idx, &sub_mem));
      RT_CHECK(topsMemoryGetDims(
          sub_mem, reinterpret_cast<int64_t *>(dims.data()), &ndims));
      ss << "    sub " << idx << " " << sub_mem << ": "
         << VectorToString(
                std::vector<int64_t>(dims.begin(), dims.begin() + ndims))
         << "\n";
    }
    ss << "    }\n";
  } else {
    ss << "  nullptr\n";
  }
  return ss.str();
}

std::string HLIRTensorToString(const std::vector<hlir::Tensor *> &tensors,
                               bool is_inputs) {
  const auto numel = tensors.size();
  std::stringstream ss;
  hlir::Tensor *tensor = nullptr;
  const std::string name = is_inputs ? "input_" : "output_";
  for (size_t i = 0; i < numel; i++) {
    tensor = tensors[i];
    ss << "tensor " << name << i << ": {\n";
    ss << "  mem_handle: {\n" << DumpHbm(tensor->mem_handle);
    ss << "  }\n";
    ss << "  bytes_size: " << tensor->bytes_size << "\n";
    ss << "  element_type: " << tensor->element_type << "\n";
    ss << "  dimensions: " << VectorToString(tensor->dimensions) << "\n";
    ss << "  strides: " << VectorToString(tensor->strides) << "\n";
    ss << "  layouts: " << VectorToString(tensor->layout) << "\n";
    ss << "}\n";
  }
  return ss.str();
}

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
