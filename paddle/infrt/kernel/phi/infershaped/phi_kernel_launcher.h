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

#include <llvm/ADT/SmallVector.h>
#include <iostream>

#include "paddle/infrt/backends/host/phi_context.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launcher.h"
#include "paddle/infrt/kernel/phi/infershaped/infershaped_utils.h"

namespace infrt {
namespace kernel {

template <typename F>
struct FuncArgStatics {};

template <typename Return, typename... Args>
struct FuncArgStatics<Return (*)(Args...)> {
  constexpr static int arg_size = sizeof...(Args);
};

template <typename KernelFunc,
          KernelFunc kernel,
          typename InferShapedFunc,
          InferShapedFunc infershape>
void KernelLauncherFunc(host_context::KernelFrame* frame) {
  static InferShapedKernelLauncher launcher(
      FuncArgStatics<InferShapedFunc>::arg_size);
  static const uint16_t num_input_tensors{InferShapeHelper<KernelFunc>::count};
  static const bool turn_on_infer_shape_cache{true};

#ifndef NDEBUG
  LOG(INFO) << "Kernel.frame: " << frame->DumpArgTypes();
#endif
  // Build the infershape KernelFrame if needed.
  // TODO(Superjomn) add unlikely here.
  if (launcher.infershape_kernel_frame_builder.IsEmpty()) {
    launcher.CreateKernelFrameForInferShape(frame);
#ifndef NDEBUG
    LOG(INFO) << "infershape.frame: "
              << launcher.infershape_kernel_frame_builder.DumpArgTypes();
#endif
  }
  if (turn_on_infer_shape_cache) {
    if (launcher.IsShapeChanged(num_input_tensors)) {
      ::infrt::host_context::KernelImpl<InferShapedFunc, infershape>::Invoke(
          &launcher.infershape_kernel_frame_builder);
      launcher.BuildInferShapeCache(num_input_tensors);
    }
  }
  ::infrt::host_context::KernelImpl<KernelFunc, kernel>::Invoke(frame);
}

}  // namespace kernel
}  // namespace infrt
