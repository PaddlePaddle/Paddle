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

#include "paddle/phi/kernels/gather_kernel.h"

#include <sys/syscall.h>
#include <sys/types.h>
#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#define gettid() syscall(__NR_gettid)

namespace phi {

template <typename T, typename Context>
void GatherKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& index,
                  const Scalar& axis,
                  DenseTensor* out) {
  auto axis_v = axis.to<int>();
  const auto& index_type = index.dtype();

  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0) return;

  const auto index_dims = index.dims();
  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(
        index_dims[1],
        1,
        phi::errors::InvalidArgument(
            "The last dim of index should be 1 when it is 2D, but we get %d",
            index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        index_dims.size() == 1 || index_dims.size() == 0,
        true,
        phi::errors::InvalidArgument(
            "The index should be 0D, 1D, when it is not 2D, but we get %d",
            index_dims.size()));
  }
  std::vector<int> xshape(x.dims().size());
  for (int i = 0; i < x.dims().size(); ++i) {
    xshape[i] = x.dims()[i];
  }
#if 0
  if ((out->dims().size() == 1 && out->dims()[0] <= 4) || index.dims().size() == 1) { // NOLINT
    dev_ctx.Wait();
    xpu_wait();
    {
      DenseTensor x_cpu(x.type());
      phi::Copy(dev_ctx, x, phi::CPUPlace(), true, &x_cpu);
      std::stringstream os;
      for (size_t i = 0; i < x.numel() && i < 10; i++) {
        os << x_cpu.data<T>()[i] << ",";
      }
      LOG(INFO) << "gather " << " tid=" << gettid() << " x_dims=" << x.dims() << " x_type=" << typeid(T).name() << " x_ptr=" << x.data<T>() << " x_data=[" << os.str() << "]"; // NOLINT
    }
    {
      DenseTensor index_cpu(index.type());
      phi::Copy(dev_ctx, index, phi::CPUPlace(), true, &index_cpu);
      std::stringstream os;
      if (index_type == DataType::INT32) {
        for (size_t i = 0; i < index.numel() && i < 10; i++) {
          os << index_cpu.data<int>()[i] << ",";
        }
        LOG(INFO) << "gather " << " tid=" << gettid() << " index_dims=" << index.dims() << "(" << index.numel()<< ") index_type="<< index_type << " index_ptr=" << index.data<int>() << " index_data=[" << os.str() << "]"; // NOLINT
      } else if (index_type == DataType::INT64) {
        for (size_t i = 0; i < index.numel() && i < 10; i++) {
          os << index_cpu.data<int64_t>()[i] << ",";
        }
        LOG(INFO) << "gather " << " tid=" << gettid() << " index_dims=" << index.dims() << "(" << index.numel()<< ") index_type="<< index_type << " index_ptr=" << index.data<int64_t>() << " index_data=[" << os.str() << "]"; // NOLINT
      } else {
        LOG(INFO) << "unknown index type " << index_type;
        exit(-1);
      }
    }
  }
#endif
  using XPUType = typename XPUTypeTrait<T>::Type;

  int r = XPU_SUCCESS;
  if (index_type == DataType::INT32) {
    r = xpu::gather<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index.data<int>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        xshape,
        index.dims().size() == 0 ? 1 : index.dims()[0],
        axis_v);
  } else if (index_type == DataType::INT64) {
    r = xpu::gather<XPUType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index.data<int64_t>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        xshape,
        index.dims().size() == 0 ? 1 : index.dims()[0],
        axis_v);
  } else {
    LOG(INFO) << "unknown index type " << index_type;
    exit(-1);
  }
  PADDLE_ENFORCE_EQ(
      r,
      xpu::Error_t::SUCCESS,
      phi::errors::External(
          "XPU gather kernel return wrong value[%d %s]", r, XPUAPIErrorMsg[r]));
#if 0
  if ((out->dims().size() == 1 && out->dims()[0] <= 4) || index.dims().size() == 1) { // NOLINT
    dev_ctx.Wait();
    xpu_wait();
    DenseTensor out_cpu(out->type());
    phi::Copy(dev_ctx, *out, phi::CPUPlace(), true, &out_cpu);
    std::stringstream os;
    for (size_t i = 0; i < out_cpu.numel() && i < 10; i++) {
      os << out_cpu.data<T>()[i] << ",";
    }
    LOG(INFO) << "gather " << " tid=" << gettid()  << " axis=" << axis_v << " out_dims=" << out->dims() << " out_type=" << typeid(T).name() << " out_ptr=" << out->data<T>() << " out_data=[" << os.str() << "]"; // NOLINT
  }
#endif
#if 0
  dev_ctx.Wait();
  LOG(INFO) << "check gather tid=" << gettid();
  phi::backends::xpu::xpu_mem_check(out->data<T>(), sizeof(T) * product(out->dims())); // NOLINT
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(gather,
                   XPU,
                   ALL_LAYOUT,
                   phi::GatherKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   bool) {}
