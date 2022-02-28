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

static void FakePhiInferShape(const ::phi::MetaTensor& a,
                              const ::phi::MetaTensor& b,
                              bool arg_0,
                              bool arg_1,
                              ::phi::MetaTensor* c) {
  LOG(INFO) << "the ptr of c: " << c;
  LOG(INFO) << "c->numel(): " << c->numel();
}

static void FakePhiKernel(const ::phi::CPUContext& /*Context*/,
                          const ::phi::DenseTensor& a,
                          const ::phi::DenseTensor& b,
                          bool arg_0,
                          bool arg_1,
                          ::phi::DenseTensor* c) {
  std::cout << "@FakePhiKernel@" << std::endl;
  LOG(INFO) << "the ptr of c: " << c;
  LOG(INFO) << "c->numel(): " << c->numel();
}

template <typename KernelFunc,
          KernelFunc kernel,
          typename InferShapedFunc,
          InferShapedFunc infershape>
class KernelLauncher : public InferShapedKernelLauncher {
 public:
  static const uint16_t num_input_tensors{InferShapeHelper<KernelFunc>::count};
  static const bool turn_on_infer_shape_cache{true};
  void Invoke(host_context::KernelFrame* frame) override {
#ifndef NDEBUG
    LOG(INFO) << "Kernel.frame: " << frame->DumpArgTypes();
#endif
    // Build the infershape KernelFrame if needed.
    // TODO(Superjomn) add unlikely here.
    if (infershape_kernel_frame_builder.IsEmpty()) {
      CreateKernelFrameForInferShape(frame);
#ifndef NDEBUG
      LOG(INFO) << "infershape.frame: "
                << infershape_kernel_frame_builder.DumpArgTypes();
#endif
    }
    if (turn_on_infer_shape_cache) {
      if (!turn_on_infer_shape_cache || IsShapeChanged(num_input_tensors)) {
        ::infrt::host_context::KernelImpl<InferShapedFunc, infershape>::Invoke(
            &infershape_kernel_frame_builder);
        BuildInferShapeCache(num_input_tensors);
      }
    }
    ::infrt::host_context::KernelImpl<KernelFunc, kernel>::Invoke(frame);
  }
};

template <typename KernelFunc,
          KernelFunc kernel,
          typename InferShapedFunc,
          InferShapedFunc infershape>
void KernelLauncherFunc(
    KernelLauncher<KernelFunc, kernel, InferShapedFunc, infershape> launcher,
    host_context::KernelFrame* frame) {
  launcher.Invoke(frame);
}

}  // namespace kernel
}  // namespace infrt
