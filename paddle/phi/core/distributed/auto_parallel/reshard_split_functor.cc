// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/auto_parallel/reshard_split_functor.h"
#include "glog/logging.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {
namespace distributed {
namespace auto_parallel {

ReshardSplitFunctor::ReshardSplitFunctor(const phi::KernelKey& kernel_key,
                                         const IntArray& sections,
                                         int64_t axis)
    : sections_(sections), axis_(axis) {
  KernelResult kernel_result =
      phi::KernelFactory::Instance().SelectKernelOrThrowError("split",
                                                              kernel_key);
  const Kernel& kernel = kernel_result.kernel;
  VLOG(3) << "Select split kernel: " << kernel;
  functor_ = kernel.GetVariadicKernelFn<SPLIT_KERNEL_SIG>();
}

void ReshardSplitFunctor::operator()(const DeviceContext& dev_ctx,
                                     const DenseTensor& input,
                                     std::vector<DenseTensor>* output) {
  std::vector<DenseTensor*> out_ptr_vec;
  for (size_t i = 0; i < output->size(); ++i) {
    out_ptr_vec.emplace_back(&(output->at(i)));
  }
  PrepareOutput(input, out_ptr_vec);
  (*functor_)(dev_ctx, input, sections_, axis_, out_ptr_vec);
}

void ReshardSplitFunctor::PrepareOutput(
    const DenseTensor& input, const std::vector<DenseTensor*>& output) {
  auto out_meta_vec = paddle::experimental::MakeMetaTensor(output);

  std::vector<phi::MetaTensor*> out_metas(out_meta_vec.size());
  for (size_t i = 0; i < out_meta_vec.size(); ++i) {
    out_metas[i] = output[i] ? &out_meta_vec[i] : nullptr;
  }

  phi::SplitInferMeta(
      paddle::experimental::MakeMetaTensor(input), sections_, axis_, out_metas);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
