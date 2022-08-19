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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/kernels/isfinite_kernel.h"
#include "paddle/phi/kernels/reduce_all_kernel.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
// store the result bool in gpu tensor, async operation. Faster than above ones.
void TensorContainsNAN(const framework::Tensor& tensor, framework::Tensor* out);
void TensorContainsInf(const framework::Tensor& tensor, framework::Tensor* out);
void TensorIsfinite(const framework::Tensor& tensor, framework::Tensor* out);

// copy the result bool to cpu
bool TensorContainsNAN(const framework::Tensor& tensor);
bool TensorContainsInf(const framework::Tensor& tensor);
bool TensorIsfinite(const framework::Tensor& tensor);

#define FiniteVisitor(type)                                                  \
  struct type##Visitor {                                                     \
    type##Visitor(const phi::DenseTensor& in, phi::DenseTensor* out)         \
        : in_(in), out_(out) {}                                              \
    template <typename T>                                                    \
    void apply() const {                                                     \
      auto place = in_.place();                                              \
      auto* ctx = platform::DeviceContextPool::Instance().Get(place);        \
      Tensor tmp;                                                            \
      tmp.Resize(in_.dims());                                                \
      out_->Resize({1});                                                     \
      std::vector<int64_t> dims(tmp.dims().size());                          \
      std::iota(dims.begin(), dims.end(), 0);                                \
      if (platform::is_cpu_place(place)) {                                   \
        phi::type##Kernel<T, phi::CPUContext>(                               \
            *static_cast<phi::CPUContext*>(ctx), in_, &tmp);                 \
        phi::AllKernel<bool, phi::CPUContext>(                               \
            *static_cast<phi::CPUContext*>(ctx), tmp, dims, false, out_);    \
      } else if (platform::is_gpu_place(place)) {                            \
        phi::type##Kernel<T, phi::GPUContext>(                               \
            *static_cast<phi::GPUContext*>(ctx), in_, &tmp);                 \
        phi::AllKernel<bool, phi::GPUContext>(                               \
            *static_cast<phi::GPUContext*>(ctx), tmp, dims, false, out_);    \
      } else {                                                               \
        PADDLE_THROW(                                                        \
            platform::errors::Unimplemented("Not supported on %s.", place)); \
      }                                                                      \
    }                                                                        \
    const phi::DenseTensor& in_;                                             \
    phi::DenseTensor* out_;                                                  \
  };

FiniteVisitor(Isnan);
FiniteVisitor(Isinf);
FiniteVisitor(Isfinite);

// store the result bool in gpu tensor, async operation. Faster than above ones.
inline void TensorContainsNAN(const framework::Tensor& tensor,
                              framework::Tensor* out) {
  VisitDataTypeNormal(TransToProtoVarType(tensor.dtype()),
                      IsnanVisitor(tensor, out));
}
inline void TensorContainsInf(const framework::Tensor& tensor,
                              framework::Tensor* out) {
  VisitDataTypeNormal(TransToProtoVarType(tensor.dtype()),
                      IsinfVisitor(tensor, out));
}
inline void TensorIsfinite(const framework::Tensor& tensor,
                           framework::Tensor* out) {
  VisitDataTypeNormal(TransToProtoVarType(tensor.dtype()),
                      IsfiniteVisitor(tensor, out));
}

// copy the result bool to cpu
inline bool TensorContainsNAN(const framework::Tensor& tensor) {
  Tensor out;
  TensorContainsNAN(tensor, &out);
  return GetValue<bool>(&out);
}
inline bool TensorContainsInf(const framework::Tensor& tensor) {
  Tensor out;
  TensorContainsInf(tensor, &out);
  return GetValue<bool>(&out);
}
inline bool TensorIsfinite(const framework::Tensor& tensor) {
  Tensor out;
  TensorIsfinite(tensor, &out);
  return GetValue<bool>(&out);
}
}  // namespace framework
namespace operators {
struct InfinityFunctor {
  void operator()(const framework::Tensor& tensor, framework::Tensor* out) {
    framework::TensorContainsInf(tensor, out);
  }
};

struct NANFunctor {
  void operator()(const framework::Tensor& tensor, framework::Tensor* out) {
    framework::TensorContainsNAN(tensor, out);
  }
};

struct IsfiniteFunctor {
  void operator()(const framework::Tensor& tensor, framework::Tensor* out) {
    framework::TensorIsfinite(tensor, out);
  }
};

template <typename DeviceContext, typename T, typename Functor>
class OverflowKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* x = ctx.InputVar("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    Functor functor;
    if (x->IsType<framework::LoDTensor>()) {
      auto* in = ctx.Input<framework::Tensor>("X");
      functor(*in, out);
    } else if (x->IsType<phi::SelectedRows>()) {
      auto& in = ctx.Input<phi::SelectedRows>("X")->value();
      functor(in, out);
    } else {
      PADDLE_ENFORCE_EQ(
          true,
          false,
          platform::errors::InvalidArgument(
              "The input type mismatch, the type of Input(X) must be Tensor or "
              "SelectedRows, please check your input."));
    }
  }
};

}  // namespace operators
}  // namespace paddle
