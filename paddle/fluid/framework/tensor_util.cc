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

#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace framework {
template <typename Predicate, typename DevCtx>
struct AnyDTypeVisitor {
  Predicate predicate_;
  const Tensor& tensor_;
  const DevCtx& ctx_;
  Tensor* out_;

  AnyDTypeVisitor(Predicate predicate, const Tensor& tensor, const DevCtx& ctx,
                  Tensor* out)
      : predicate_(predicate), tensor_(tensor), ctx_(ctx), out_(out) {}

  template <typename T>
  void operator()() const {
    auto t = EigenVector<T>::Flatten(tensor_);
    auto o = EigenScalar<bool>::From(*out_);
    // return any of predicate_(t) is true.
    o.device(*ctx_.eigen_device()) = predicate_(t).any();
  }
};

template <typename Predicate, typename DevCtx>
inline void AnyImpl(Predicate predicate, const framework::Tensor& tensor,
                    const DevCtx& ctx, framework::Tensor* out) {
  VisitDataType(ToDataType(tensor.type()), AnyDTypeVisitor<Predicate, DevCtx>(
                                               predicate, tensor, ctx, out));
}

template <typename Predicate>
struct AnyVisitor : public boost::static_visitor<bool> {
  const framework::Tensor& tensor_;
  Predicate predicate_;

  AnyVisitor(const framework::Tensor& tensor, Predicate predicate)
      : tensor_(tensor), predicate_(std::move(predicate)) {}

  template <typename Place>
  bool operator()(const Place& place) const {
    framework::Tensor out;
    out.Resize({1});
    out.mutable_data<bool>(place);
    auto* ctx = platform::DeviceContextPool::Instance().GetByPlace(place);
    AnyImpl(predicate_, tensor_, *ctx, &out);
    return this->GetResult(out, place);
  }

  bool GetResult(const framework::Tensor& out,
                 const platform::CUDAPlace& gpu) const {
    platform::CPUPlace cpu;
    framework::Tensor tmp;
    tmp.Resize({1});
    tmp.mutable_data<bool>(cpu);
    auto gpuctx = platform::DeviceContextPool::Instance().Get(gpu);
    gpuctx->Wait();
    Copy(out, cpu, *gpuctx, &tmp);
    gpuctx->Wait();
    return GetResult(tmp, cpu);
  }

  bool GetResult(const framework::Tensor& out,
                 const platform::CPUPlace& cpu) const {
    return *out.data<bool>();
  }
};

template <typename Predicate>
inline bool Any(const framework::Tensor& tensor, Predicate predicate) {
  AnyVisitor<Predicate> visitor(tensor, predicate);
  auto place = tensor.place();
  return platform::VisitPlace(place, visitor);
}

struct HasNANPredicate {
  template <typename T>
  auto operator()(const T& eigen_vec) const
      -> decltype(std::declval<T>().isnan()) {
    // Cast eigen_vector to vector of bool. true if is inf.
    return eigen_vec.isnan();
  }
};

bool HasNAN(const framework::Tensor& tensor) {
  HasNANPredicate predicate;
  return Any(tensor, predicate);
}

struct HasInfPredicate {
  template <typename T>
  auto operator()(const T& eigen_vec) const
      -> decltype(std::declval<T>().isinf()) {
    // Cast eigen_vector to vector of bool. true if is inf.
    return eigen_vec.isinf();
  }
};

bool HasInf(const framework::Tensor& tensor) {
  HasInfPredicate predicate;
  return Any(tensor, predicate);
}

}  // namespace framework
}  // namespace paddle
