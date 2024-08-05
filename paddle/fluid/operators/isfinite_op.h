// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/isfinite_kernel.h"
#include "paddle/phi/kernels/reduce_all_kernel.h"
#include "paddle/phi/kernels/reduce_any_kernel.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
// store the result bool in gpu tensor, async operation. Faster than above ones.
void TensorContainsNAN(const phi::DenseTensor& tensor, phi::DenseTensor* out);
void TensorContainsInf(const phi::DenseTensor& tensor, phi::DenseTensor* out);
void TensorIsfinite(const phi::DenseTensor& tensor, phi::DenseTensor* out);

// copy the result bool to cpu
bool TensorContainsNAN(const phi::DenseTensor& tensor);
bool TensorContainsInf(const phi::DenseTensor& tensor);
bool TensorIsfinite(const phi::DenseTensor& tensor);

#define FiniteVisitor(type, reduce_type, device)                             \
  struct type##Visitor##device {                                             \
    type##Visitor##device(const phi::DenseTensor& in, phi::DenseTensor* out) \
        : in_(in), out_(out) {}                                              \
    template <typename T>                                                    \
    void apply() const {                                                     \
      auto place = in_.place();                                              \
      auto* ctx = static_cast<phi::device##Context*>(                        \
          phi::DeviceContextPool::Instance().Get(place));                    \
      phi::DenseTensor tmp;                                                  \
      tmp.Resize(in_.dims());                                                \
      out_->Resize({1});                                                     \
      std::vector<int64_t> dims(tmp.dims().size());                          \
      std::iota(dims.begin(), dims.end(), 0);                                \
      phi::type##Kernel<T, phi::device##Context>(*ctx, in_, &tmp);           \
      phi::reduce_type##Kernel<bool, phi::device##Context>(                  \
          *ctx, tmp, dims, false, out_);                                     \
    }                                                                        \
    const phi::DenseTensor& in_;                                             \
    phi::DenseTensor* out_;                                                  \
  };

FiniteVisitor(Isnan, Any, CPU);
FiniteVisitor(Isinf, Any, CPU);
FiniteVisitor(Isfinite, All, CPU);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
FiniteVisitor(Isnan, Any, GPU);
FiniteVisitor(Isinf, Any, GPU);
FiniteVisitor(Isfinite, All, GPU);
#endif

// store the result bool in gpu tensor, async operation. Faster than above ones.
inline void TensorContainsNAN(const phi::DenseTensor& tensor,
                              phi::DenseTensor* out) {
  auto place = tensor.place();
  if (tensor.place().GetType() == phi::AllocationType::CPU) {
    VisitDataTypeNormal(paddle::framework::TransToProtoVarType(tensor.dtype()),
                        IsnanVisitorCPU(tensor, out));
    return;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (place.GetType() == phi::AllocationType::GPU) {
    VisitDataTypeNormal(paddle::framework::TransToProtoVarType(tensor.dtype()),
                        IsnanVisitorGPU(tensor, out));
    return;
  }
#endif
  PADDLE_THROW(common::errors::Unimplemented("Not supported on %s.", place));
}
inline void TensorContainsInf(const phi::DenseTensor& tensor,
                              phi::DenseTensor* out) {
  auto place = tensor.place();
  if (tensor.place().GetType() == phi::AllocationType::CPU) {
    VisitDataTypeNormal(paddle::framework::TransToProtoVarType(tensor.dtype()),
                        IsinfVisitorCPU(tensor, out));
    return;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (place.GetType() == phi::AllocationType::GPU) {
    VisitDataTypeNormal(paddle::framework::TransToProtoVarType(tensor.dtype()),
                        IsinfVisitorGPU(tensor, out));
    return;
  }
#endif
  PADDLE_THROW(common::errors::Unimplemented("Not supported on %s.", place));
}
inline void TensorIsfinite(const phi::DenseTensor& tensor,
                           phi::DenseTensor* out) {
  auto place = tensor.place();
  if (tensor.place().GetType() == phi::AllocationType::CPU) {
    VisitDataTypeNormal(paddle::framework::TransToProtoVarType(tensor.dtype()),
                        IsfiniteVisitorCPU(tensor, out));
    return;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (place.GetType() == phi::AllocationType::GPU) {
    VisitDataTypeNormal(paddle::framework::TransToProtoVarType(tensor.dtype()),
                        IsfiniteVisitorGPU(tensor, out));
    return;
  }
#endif
  PADDLE_THROW(common::errors::Unimplemented("Not supported on %s.", place));
}

// copy the result bool to cpu
inline bool TensorContainsNAN(const phi::DenseTensor& tensor) {
  phi::DenseTensor out;
  TensorContainsNAN(tensor, &out);
  return paddle::framework::GetValue<bool>(&out);
}
inline bool TensorContainsInf(const phi::DenseTensor& tensor) {
  phi::DenseTensor out;
  TensorContainsInf(tensor, &out);
  return paddle::framework::GetValue<bool>(&out);
}
inline bool TensorIsfinite(const phi::DenseTensor& tensor) {
  phi::DenseTensor out;
  TensorIsfinite(tensor, &out);
  return paddle::framework::GetValue<bool>(&out);
}
}  // namespace framework
namespace operators {
struct InfinityFunctor {
  void operator()(const phi::DenseTensor& tensor, phi::DenseTensor* out) {
    framework::TensorContainsInf(tensor, out);
  }
};

struct NANFunctor {
  void operator()(const phi::DenseTensor& tensor, phi::DenseTensor* out) {
    framework::TensorContainsNAN(tensor, out);
  }
};

template <typename DeviceContext, typename T, typename Functor>
class OverflowKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* x = ctx.InputVar("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    Functor functor;
    if (x->IsType<phi::DenseTensor>()) {
      auto* in = ctx.Input<phi::DenseTensor>("X");
      functor(*in, out);
    } else if (x->IsType<phi::SelectedRows>()) {
      auto& in = ctx.Input<phi::SelectedRows>("X")->value();
      functor(in, out);
    } else {
      PADDLE_ENFORCE_EQ(true,
                        false,
                        common::errors::InvalidArgument(
                            "The input type mismatch, the type of Input(X) "
                            "must be phi::DenseTensor or "
                            "SelectedRows, please check your input."));
    }
  }
};

}  // namespace operators
}  // namespace paddle
