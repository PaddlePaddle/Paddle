/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/place.h"

PADDLE_DEFINE_EXPORTED_bool(
    benchmark, false,
    "Doing memory benchmark. It will make deleting scope synchronized, "
    "and add some memory usage logs."
    "Default cuda is asynchronous device, set to True will"
    "force op run in synchronous mode.");

namespace paddle {
namespace platform {

bool is_gpu_place(const Place &p) {
  return p.GetType() == pten::AllocationType::GPU;
}

bool is_xpu_place(const Place &p) {
  return p.GetType() == pten::AllocationType::XPU;
}

bool is_mlu_place(const Place &p) {
  return p.GetType() == pten::AllocationType::MLU;
}

bool is_npu_place(const Place &p) {
  return p.GetType() == pten::AllocationType::NPU;
}

bool is_ipu_place(const Place &p) {
  return p.GetType() == pten::AllocationType::IPU;
}

bool is_cpu_place(const Place &p) {
  return p.GetType() == pten::AllocationType::CPU;
}

bool is_cuda_pinned_place(const Place &p) {
  return p.GetType() == pten::AllocationType::GPUPINNED;
}

bool is_npu_pinned_place(const Place &p) {
  return p.GetType() == pten::AllocationType::NPUPINNED;
}

bool places_are_same_class(const Place &p1, const Place &p2) {
  return p1.GetType() == p2.GetType();
}

bool is_same_place(const Place &p1, const Place &p2) {
  if (places_are_same_class(p1, p2)) {
    if (is_cpu_place(p1) || is_cuda_pinned_place(p1) ||
        is_npu_pinned_place(p1)) {
      return true;
    } else if (is_xpu_place(p1)) {
      return p1 == p2;
    } else if (is_mlu_place(p1)) {
      return p1 == p2;
    } else if (is_npu_place(p1)) {
      return p1 == p2;
    } else if (is_ipu_place(p1)) {
      return p1 == p2;
    } else {
      return p1 == p2;
    }
  } else {
    return false;
  }
}

}  // namespace platform
}  // namespace paddle
