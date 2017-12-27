/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/framework/data_type.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {

/**
 * @brief   Copy the content of external tensor to a new place.
 *
 * @param[in] src        The external tensor.
 * @param[in] dst_place  The dst place.
 * @param[in] ctx        The device context contains device resources.
 *
 * @note    CopyFrom supports CPU <-> GPU, GPU <-> GPU.
 */

inline void CopyFrom(const Tensor& src, const platform::Place& dst_place,
                     const platform::DeviceContext& ctx, Tensor* dst) {
  src.check_memory_size();

  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  auto src_place = src.place();
  auto src_ptr = src.data<void>();

  auto dst_ptr = dst->mutable_data(dst_place, src.type());

  auto size = src.numel() * SizeOfType(src.type());

  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr,
                 boost::get<platform::CPUPlace>(src_place), src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    auto src_gpu_place = boost::get<platform::CUDAPlace>(src_place);
    auto dst_cpu_place = boost::get<platform::CPUPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::CUDAPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place, ctx_gpu_place);
    memory::Copy(
        dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = boost::get<platform::CPUPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::CUDAPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::CUDAPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(dst_gpu_place, ctx_gpu_place);
    memory::Copy(
        dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  } else if (platform::is_gpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = boost::get<platform::CUDAPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::CUDAPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::CUDAPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place, ctx_gpu_place);
    memory::Copy(
        dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

/**
 * @brief CopyFrom support CPU <-> CPU
 */
inline void CopyFrom(const Tensor& src, const platform::Place& dst_place,
                     Tensor* dst) {
  src.check_memory_size();
  dst->Resize(src.dims());
  dst->set_layout(src.layout());

  auto src_place = src.place();
  auto src_ptr = src.data<void>();

  auto dst_ptr = dst->mutable_data(dst_place, src.type());

  auto size = src.numel() * SizeOfType(src.type());

  PADDLE_ENFORCE(platform::is_cpu_place(src_place) &&
                 platform::is_cpu_place(dst_place));

  memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr,
               boost::get<platform::CPUPlace>(src_place), src_ptr, size);
}

/**
 * @brief   Copy the content of an external vector to a tensor.
 *
 * @param[in] src        The external tensor.
 * @param[in] ctx        The device context contains device resources.
 *
 * * @note    CopyFromVector assumes that the tensor has been resized
 *            before invoking.
 */
template <typename T>
inline void CopyFromVector(const std::vector<T>& src,
                           const platform::DeviceContext& ctx, Tensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  if (platform::is_cpu_place(dst_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr, src_place,
                 src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(
        boost::get<platform::CUDAPlace>(dst_place), dst_ptr, src_place, src_ptr,
        size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

/**
 * @brief CopyFromVector CPU vector -> CPU Tensor
 */
template <typename T>
inline void CopyFromVector(const std::vector<T>& src, Tensor* dst) {
  platform::CPUPlace dst_place = platform::CPUPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
}

/**
 * @brief   Copy the content of a tensor to a vector
 *
 * @param[in] src        The external tensor.
 * @param[in] ctx        The device context contains device resources.
 *
 * * @note    CopyFromVector assumes that the tensor has been resized
 *            before invoking.
 */
template <typename T>
inline void CopyToVector(const Tensor& src, const platform::DeviceContext& ctx,
                         std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  if (platform::is_cpu_place(src.place())) {
    memory::Copy(dst_place, dst_ptr,
                 boost::get<platform::CPUPlace>(src.place()), src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src.place())) {  // NOLINT
    memory::Copy(
        dst_place, dst_ptr, boost::get<platform::CUDAPlace>(src.place()),
        src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

/**
 * @brief CopyToVector CPUTensor <-> CPU Vector
 */
template <typename T>
inline void CopyToVector(const Tensor& src, std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  PADDLE_ENFORCE(platform::is_cpu_place(src.place()));

  memory::Copy(dst_place, dst_ptr, boost::get<platform::CPUPlace>(src.place()),
               src_ptr, size);
}

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
    platform::DeviceContextPool::Instance().Get(gpu)->Wait();
    CopyFrom(out, cpu, &tmp);
    platform::DeviceContextPool::Instance().Get(gpu)->Wait();
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

struct HasNanPredicate {
  template <typename T>
  auto operator()(T eigen_vec) const -> decltype(std::declval<T>().isnan()) {
    return eigen_vec.isnan();
  }
};

inline bool HasNan(const framework::Tensor& tensor) {
  HasNanPredicate predicate;
  return Any(tensor, predicate);
}

struct HasInfPredicate {
  template <typename T>
  auto operator()(T eigen_vec) const -> decltype(std::declval<T>().isinf()) {
    return eigen_vec.isinf();
  }
};

inline bool HasInf(const framework::Tensor& tensor) {
  HasInfPredicate predicate;
  return Any(tensor, predicate);
}

}  // namespace framework
}  // namespace paddle
