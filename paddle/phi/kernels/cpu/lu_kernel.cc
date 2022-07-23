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

#include "paddle/phi/kernels/lu_kernel.h"

namespace phi {

template <typename T, typename Context>
void LUKernel(const Context& dev_ctx,
    const DenseTensor& x,
    bool pivot,
    DenseTensor* out,
    DenseTensor* pivots,
    DenseTensor* infos){
  auto pivots = ctx.Attr<bool>("pivots");
  auto *xin = ctx.Input<framework::Tensor>("X");
  auto *out = ctx.Output<framework::Tensor>("Out");
  auto *IpivT = ctx.Output<framework::Tensor>("Pivots");
  auto *InfoT = ctx.Output<framework::Tensor>("Infos");
  PADDLE_ENFORCE_EQ(pivots,
                      true,
                      platform::errors::InvalidArgument(
                          "lu without pivoting is not implemented on the CPU, "
                          "but got pivots=False"));

  math::DeviceIndependenceTensorOperations<phi::CPUContext, T> helper(ctx);
  *out = helper.Transpose(*xin);

  auto outdims = out->dims();
  auto outrank = outdims.size();

  int m = static_cast<int>(outdims[outrank - 1]);
  int n = static_cast<int>(outdims[outrank - 2]);
  int lda = std::max(1, m);

  auto ipiv_dims = phi::slice_ddim(outdims, 0, outrank - 1);
  ipiv_dims[outrank - 2] = std::min(m, n);
  IpivT->Resize(ipiv_dims);
  auto ipiv_data = IpivT->mutable_data<int>(ctx.GetPlace());

  auto info_dims = phi::slice_ddim(outdims, 0, outrank - 2);
  if (info_dims.size() == 0) {
    info_dims = phi::make_ddim({1});
  }
  InfoT->Resize(info_dims);
  auto info_data = InfoT->mutable_data<int>(ctx.GetPlace());

  auto batchsize = product(info_dims);
  batchsize = std::max(static_cast<int>(batchsize), 1);
  auto out_data = out->mutable_data<T>(ctx.GetPlace());
  for (int b = 0; b < batchsize; b++) {
    auto out_data_item = &out_data[b * m * n];
    int *info_data_item = &info_data[b];
    int *ipiv_data_item = &ipiv_data[b * std::min(m, n)];
    phi::funcs::lapackLu<T>(
        m, n, out_data_item, lda, ipiv_data_item, info_data_item);
  }
  *out = helper.Transpose(*out);
}

}  // namespace phi