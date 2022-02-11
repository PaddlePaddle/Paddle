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

#include "paddle/infrt/host_context/kernel_frame.h"
#include "paddle/infrt/host_context/value.h"
#include "paddle/infrt/naive/meta_tensor.h"
#include "paddle/infrt/tensor/dense_host_tensor.h"

namespace infrt {
namespace naive {

struct InferShapedKernelLauncher {
  virtual void Invoke(host_context::KernelFrame* frame) = 0;

  virtual ~InferShapedKernelLauncher() = default;

 protected:
  //! Initialize the kernel frame for InferShape kernel.
  // This method will create a new KernelFrame with all the Tensors(currently
  // only DenseHostTensor) converted into MetaTensors so that the infer-shape
  // function can work with.
  // @frame: the frame containing argument list that is same with the ones of
  // the corresponding kernel.
  void CreateKernelFrameForInferShape(host_context::KernelFrame* frame);

  //! Build or update the infer-shape cache using the latest shape from
  //! InferShapeFrame.
  void BuildInferShapeCache(const uint16_t num_inputs);

  //! Compare the latest shape with the shape cache.
  bool IsShapeChanged(const uint16_t num_inputs) const;

  // values to hold the TensorMeta.
  llvm::SmallVector<host_context::ValueRef, 3> values;
  llvm::SmallVector<tensor::TensorShape, 3> tensor_shape_cache;
  host_context::KernelFrameBuilder infershape_kernel_frame_builder;
};

}  // namespace naive
}  // namespace infrt
