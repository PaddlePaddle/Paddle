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

#include "paddle/phi/kernels/all_to_all_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void AllToAllKernel(const Context& dev_ctx UNUSED,
                    const DenseTensor& x UNUSED,
                    DenseTensor* out UNUSED) {
  PADDLE_THROW(
      errors::Unimplemented("Unimplemented cpu kernel for all_to_all."));
}
#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <typename T>
void AllToAllKernel(const phi::CustomContext& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);
  auto comm_ctx =
      static_cast<distributed::XCCLCommContext*>(dev_ctx.GetCommContext());

  int nranks = comm_ctx->GetSize();
  int rank = comm_ctx->GetRank();
  int send_numel = x.numel() / nranks;

  std::vector<void*> sendbuf, recvbuf;
  std::vector<size_t> sendsize(send_numel, nranks);
  std::vector<phi::DataType> sendtype(x.dtype(), nranks);
  for (auto i = 0; i < nranks; ++i) {
    sendbuf.push_back(x.data<T>() + i * send_numel);
    recvbuf.push_back(out->data<T>() + i * send_numel);
  }
  phi::DeviceManager::CCLAllToAll(dev_ctx.GetPlace().GetDeviceType(),
                                  const_cast<const void**>(sendbuf.data()),
                                  sendsize.data(),
                                  sendtype.data(),
                                  recvbuf.data(),
                                  sendsize.data(),
                                  sendtype.data(),
                                  rank,
                                  nranks,
                                  comm_ctx->GetXcclComm(),
                                  *dev_ctx.GetStream());
}

#endif

}  // namespace phi

PD_REGISTER_KERNEL(all_to_all,
                   CPU,
                   ALL_LAYOUT,
                   phi::AllToAllKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16) {}
#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(all_to_all,
                   Custom,
                   ALL_LAYOUT,
                   phi::AllToAllKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   int16_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}
#endif
