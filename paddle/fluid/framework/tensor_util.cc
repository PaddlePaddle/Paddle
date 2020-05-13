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
#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {

void TensorCopy(const Tensor& src, const platform::Place& dst_place,
                const platform::DeviceContext& ctx, Tensor* dst) {
  if (&src == dst) {
    auto src_copy = src;
    TensorCopy(src_copy, dst_place, ctx, dst);
    return;
  }

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;
  src.check_memory_size();

  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  auto src_place = src.place();
  auto src_ptr = src.data<void>();
  auto dst_ptr = dst->mutable_data(dst_place, src.type());

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << dst_place;
    return;
  }

  auto size = src.numel() * SizeOfType(src.type());

  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    memory::Copy(BOOST_GET_CONST(platform::CPUPlace, dst_place), dst_ptr,
                 BOOST_GET_CONST(platform::CPUPlace, src_place), src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    auto src_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, src_place);
    auto dst_cpu_place = BOOST_GET_CONST(platform::CPUPlace, dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx_place), true);
    auto ctx_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place, ctx_gpu_place);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = BOOST_GET_CONST(platform::CPUPlace, src_place);
    auto dst_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx_place), true);
    auto ctx_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, ctx_place);
    PADDLE_ENFORCE_EQ(dst_gpu_place, ctx_gpu_place);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);
  } else if (platform::is_gpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, src_place);
    auto dst_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx_place), true);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    if (platform::is_same_place(src_place, dst_place)) {
      memory::Copy(dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size,
                   stream);
    } else {
      if (platform::is_same_place(ctx_place, src_place)) {
        memory::Copy(dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size,
                     stream);
        platform::DeviceContextPool::Instance().Get(src.place())->Wait();
      } else if (platform::is_same_place(ctx_place, dst_place)) {
        platform::DeviceContextPool::Instance().Get(src.place())->Wait();
        memory::Copy(dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size,
                     stream);
      } else {
        PADDLE_THROW("ctx is not belong to dst_gpu_place or src_gpu_place.");
      }
    }
  } else {
    PADDLE_THROW("Copy from %s to %s is not supported.", src_place, dst_place);
  }
#endif
}

void TensorCopy(const Tensor& src, const platform::Place& dst_place,
                Tensor* dst) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  const platform::DeviceContext* dev_ctx;
  if (platform::is_gpu_place(dst_place)) {
    dev_ctx = pool.Get(dst_place);
  } else {
    dev_ctx = pool.Get(src.place());
  }
  TensorCopy(src, dst_place, *dev_ctx, dst);
}

void TensorCopySync(const Tensor& src, const platform::Place& dst_place,
                    Tensor* dst) {
  if (&src == dst) {
    auto src_copy = src;
    TensorCopySync(src_copy, dst_place, dst);
    return;
  }

  VLOG(3) << "TensorCopySync " << src.dims() << " from " << src.place()
          << " to " << dst_place;
  src.check_memory_size();
  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  auto src_place = src.place();
  auto src_ptr = src.data<void>();
  auto dst_ptr = dst->mutable_data(dst_place, src.type());

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data from " << src_place << " to "
            << dst_place;
    return;
  }

  auto size = src.numel() * SizeOfType(src.type());
  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    memory::Copy(BOOST_GET_CONST(platform::CPUPlace, dst_place), dst_ptr,
                 BOOST_GET_CONST(platform::CPUPlace, src_place), src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    auto src_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, src_place);
    auto dst_cpu_place = BOOST_GET_CONST(platform::CPUPlace, dst_place);
    memory::Copy(dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, nullptr);
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = BOOST_GET_CONST(platform::CPUPlace, src_place);
    auto dst_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, dst_place);
    memory::Copy(dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, nullptr);
  } else if (platform::is_gpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, src_place);
    auto dst_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, dst_place);
    memory::Copy(dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, nullptr);
  } else if (platform::is_cuda_pinned_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_pinned_place =
        BOOST_GET_CONST(platform::CUDAPinnedPlace, src_place);
    auto dst_gpu_place = BOOST_GET_CONST(platform::CUDAPlace, dst_place);
    memory::Copy(dst_gpu_place, dst_ptr, src_pinned_place, src_ptr, size,
                 nullptr);
  } else {
    PADDLE_THROW("Copy from %s to %s is not supported.", src_place, dst_place);
  }
#endif
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
  void apply() const {
    auto t = EigenVector<T>::Flatten(tensor_);
    auto o = EigenScalar<bool>::From(*out_);
    // return any of predicate_(t) is true.
    o.device(*ctx_.eigen_device()) = predicate_(t).any();
  }
};

template <typename Predicate, typename DevCtx>
inline void AnyImpl(Predicate predicate, const framework::Tensor& tensor,
                    const DevCtx& ctx, framework::Tensor* out) {
  VisitDataType(tensor.type(), AnyDTypeVisitor<Predicate, DevCtx>(
                                   predicate, tensor, ctx, out));
}

template <typename Predicate>
class AnyVisitor : public boost::static_visitor<bool> {
 private:
  const framework::Tensor& tensor_;
  Predicate predicate_;

 public:
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
    TensorCopy(out, cpu, *gpuctx, &tmp);
    gpuctx->Wait();
    return GetResult(tmp, cpu);
  }

  bool GetResult(const framework::Tensor& out,
                 const platform::CPUPlace& cpu) const {
    return *out.data<bool>();
  }

  bool GetResult(const framework::Tensor& out,
                 const platform::CUDAPinnedPlace& cpu) const {
    return *out.data<bool>();
  }
};

template <typename Predicate>
class AnyOutVisitor : public boost::static_visitor<> {
 private:
  const framework::Tensor& tensor_;
  mutable framework::Tensor* out_;
  Predicate predicate_;

 public:
  AnyOutVisitor(const framework::Tensor& tensor, Predicate predicate,
                framework::Tensor* out)
      : tensor_(tensor), out_(out), predicate_(std::move(predicate)) {}

  template <typename Place>
  void operator()(const Place& place) const {
    auto* ctx = platform::DeviceContextPool::Instance().GetByPlace(place);
    out_->Resize({1});
    out_->mutable_data<bool>(place);
    AnyImpl(predicate_, tensor_, *ctx, out_);
  }
};

template <typename Predicate>
inline bool Any(const framework::Tensor& tensor, Predicate predicate) {
  AnyVisitor<Predicate> visitor(tensor, predicate);
  auto place = tensor.place();
  return platform::VisitPlace(place, visitor);
}

template <typename Predicate>
inline void Any(const framework::Tensor& tensor, Predicate predicate,
                framework::Tensor* out) {
  AnyOutVisitor<Predicate> visitor(tensor, predicate, out);
  auto place = tensor.place();
  platform::VisitPlace(place, visitor);
}

struct ContainsNANPredicate {
  template <typename T>
  auto operator()(const T& eigen_vec) const
      -> decltype(std::declval<T>().isnan()) {
    // Cast eigen_vector to vector of bool. true if is inf.
    return eigen_vec.isnan();
  }
};

bool TensorContainsNAN(const framework::Tensor& tensor) {
  ContainsNANPredicate predicate;
  return Any(tensor, predicate);
}

void TensorContainsNAN(const framework::Tensor& tensor,
                       framework::Tensor* out) {
  ContainsNANPredicate predicate;
  Any(tensor, predicate, out);
}

struct ContainsInfPredicate {
  template <typename T>
  auto operator()(const T& eigen_vec) const
      -> decltype(std::declval<T>().isinf()) {
    // Cast eigen_vector to vector of bool. true if is inf.
    return eigen_vec.isinf();
  }
};

bool TensorContainsInf(const framework::Tensor& tensor) {
  ContainsInfPredicate predicate;
  return Any(tensor, predicate);
}

void TensorContainsInf(const framework::Tensor& tensor,
                       framework::Tensor* out) {
  ContainsInfPredicate predicate;
  Any(tensor, predicate, out);
}

// NOTE(dzhwinter):
// Isfinite need a AllVisitor to loop through all the elements.
// We choose two cuda call instead of one allvisitor. The AllVisitor
// should be implemented if the performance hurts.
bool TensorIsfinite(const framework::Tensor& tensor) {
  ContainsInfPredicate pred_inf;
  ContainsNANPredicate pred_nan;
  return !Any(tensor, pred_inf) && !Any(tensor, pred_nan);
}

#ifdef PADDLE_WITH_CUDA
template <typename T>
static inline void __global__ BothFalse(const T* cmp, T* out) {
  out[0] = (!cmp[0]) && (!out[0]);
}
#endif

struct BothFalseVisitor : public boost::static_visitor<> {
  const framework::Tensor& in_;
  mutable framework::Tensor* out_;
  BothFalseVisitor(const framework::Tensor& in, framework::Tensor* out)
      : in_(in), out_(out) {}

  template <typename Place>
  void operator()(const Place& place) const {
    VisitorImpl(place);
  }

  void VisitorImpl(const platform::CUDAPlace& gpu) const {
#ifdef PADDLE_WITH_CUDA
    auto* ctx = platform::DeviceContextPool::Instance().GetByPlace(gpu);
    BothFalse<bool><<<1, 1, 0, ctx->stream()>>>(in_.data<bool>(),
                                                out_->mutable_data<bool>(gpu));
#endif
  }

  void VisitorImpl(const platform::CPUPlace& cpu) const {
    bool lhs = !in_.data<bool>()[0];
    bool rhs = !out_->mutable_data<bool>(cpu)[0];
    out_->mutable_data<bool>(cpu)[0] = lhs && rhs;
  }

  void VisitorImpl(
      const platform::CUDAPinnedPlace& cpu /* equals to cpu*/) const {
    bool lhs = !in_.data<bool>()[0];
    bool rhs = !out_->mutable_data<bool>(cpu)[0];
    out_->mutable_data<bool>(cpu)[0] = lhs && rhs;
  }
};

void TensorIsfinite(const framework::Tensor& tensor, framework::Tensor* out) {
  framework::Tensor tmp;
  TensorContainsInf(tensor, &tmp);
  TensorContainsNAN(tensor, out);
  BothFalseVisitor visitor(tmp, out);
  auto place = tensor.place();
  platform::VisitPlace(place, visitor);
}

void TensorToStream(std::ostream& os, const Tensor& tensor,
                    const platform::DeviceContext& dev_ctx) {
  {  // the 1st field, uint32_t version
    constexpr uint32_t version = 0;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  }
  {  // the 2nd field, tensor description
     // int32_t  size
     // void*    protobuf message
    proto::VarType::TensorDesc desc;
    desc.set_data_type(tensor.type());
    auto dims = framework::vectorize(tensor.dims());
    auto* pb_dims = desc.mutable_dims();
    pb_dims->Resize(static_cast<int>(dims.size()), 0);
    std::copy(dims.begin(), dims.end(), pb_dims->begin());
    int32_t size = desc.ByteSize();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    auto out = desc.SerializeAsString();
    os.write(out.data(), size);
  }
  {  // the 3rd field, tensor data
    uint64_t size = tensor.numel() * framework::SizeOfType(tensor.type());

    auto* data_ptr = tensor.data<void>();
    PADDLE_ENFORCE_LT(size, std::numeric_limits<std::streamsize>::max(),
                      platform::errors::ResourceExhausted(
                          "tensor size %d overflow when writing tensor", size));
    if (platform::is_gpu_place(tensor.place())) {
#ifdef PADDLE_WITH_CUDA
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& gpu_dev_ctx =
          static_cast<const platform::CUDADeviceContext&>(dev_ctx);
      platform::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        memory::Copy(cpu, buf.get(),
                     BOOST_GET_CONST(platform::CUDAPlace, tensor.place()),
                     reinterpret_cast<const void*>(data), size_to_write,
                     gpu_dev_ctx.stream());
        gpu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "CUDAPlace is not supported when not compiled with CUDA"));
#endif
    } else {
      os.write(static_cast<const char*>(data_ptr),
               static_cast<std::streamsize>(size));
    }
  }
}

struct DeserializedDataFunctor {
  DeserializedDataFunctor(void** buf, Tensor* tensor,
                          const platform::Place& place)
      : buf_(buf), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() {
    *buf_ = tensor_->mutable_data<T>(place_);
  }

  void** buf_;
  Tensor* tensor_;
  platform::Place place_;
};

void TensorFromStream(std::istream& is, Tensor* tensor,
                      const platform::DeviceContext& dev_ctx,
                      const size_t& seek, const std::vector<int64_t>& shape) {
  uint32_t version;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));

  PADDLE_ENFORCE_EQ(
      version, 0U,
      platform::errors::InvalidArgument(
          "tensor version %u is not supported, Only version 0 is supported",
          version));

  proto::VarType::TensorDesc desc;
  {  // int32_t size
    // proto buffer
    int32_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char*>(buf.get()), size);
    PADDLE_ENFORCE_EQ(
        desc.ParseFromArray(buf.get(), size), true,
        platform::errors::InvalidArgument("Cannot parse tensor desc"));
  }
  {  // read tensor
    tensor->Resize(framework::make_ddim(shape));
    size_t seekg = seek * framework::SizeOfType(desc.data_type());
    is.seekg(seekg, is.cur);

    void* buf;
    auto ctx = platform::CPUDeviceContext();
    size_t size = tensor->numel() * framework::SizeOfType(desc.data_type());
    if (platform::is_gpu_place(dev_ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
      Tensor cpu_tensor;
      cpu_tensor.Resize(framework::make_ddim(shape));
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, &cpu_tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);
      auto dst_place = dev_ctx.GetPlace();
      framework::TensorCopy(cpu_tensor, dst_place, dev_ctx, tensor);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "CUDAPlace is not supported when not compiled with CUDA"));
#endif
    } else {
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);
    }
  }
}

void TensorFromStream(std::istream& is, Tensor* tensor,
                      const platform::DeviceContext& dev_ctx) {
  uint32_t version;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));
  PADDLE_ENFORCE_EQ(
      version, 0U,
      platform::errors::InvalidArgument(
          "tensor version %u is not supported, Only version 0 is supported",
          version));
  proto::VarType::TensorDesc desc;
  {  // int32_t size
     // proto buffer
    int32_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char*>(buf.get()), size);
    PADDLE_ENFORCE_EQ(
        desc.ParseFromArray(buf.get(), size), true,
        platform::errors::InvalidArgument("Cannot parse tensor desc"));
  }
  {  // read tensor
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(desc.dims().size()));
    std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims));
    tensor->Resize(framework::make_ddim(dims));
    void* buf;
    auto ctx = platform::CPUDeviceContext();
    size_t size = tensor->numel() * framework::SizeOfType(desc.data_type());
    if (platform::is_gpu_place(dev_ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
      Tensor cpu_tensor;
      cpu_tensor.Resize(framework::make_ddim(dims));
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, &cpu_tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);
      auto dst_place = dev_ctx.GetPlace();
      framework::TensorCopy(cpu_tensor, dst_place, dev_ctx, tensor);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "CUDAPlace is not supported when not compiled with CUDA"));
#endif
    } else {
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);
    }
  }
}

// get tensor data point by DLDataType
void* GetDstPtrByDLDataType(DLDataType type, framework::Tensor* dst,
                            const platform::Place& dst_place) {
  // vector types not currently supported
  PADDLE_ENFORCE_LE(type.lanes, 1, "vector types not currently supported");

  switch (type.bits) {
    case 8:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int8_t>(dst_place));
      if (type.code == kDLUInt)
        return static_cast<void*>(dst->mutable_data<uint8_t>(dst_place));
      PADDLE_THROW("There is no this type.code <%d> when type.bits is <%d>.",
                   type.code, type.bits);
    case 16:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int16_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void*>(
            dst->mutable_data<paddle::platform::float16>(dst_place));
      PADDLE_THROW("There is no this type.code <%d> when type.bits is <%d>.",
                   type.code, type.bits);
    case 32:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int32_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void*>(dst->mutable_data<float>(dst_place));
      PADDLE_THROW("There is no this type.code <%d> when type.bits is <%d>.",
                   type.code, type.bits);
    case 64:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int64_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void*>(dst->mutable_data<double>(dst_place));
      PADDLE_THROW("There is no this type.code <%d> when type.bits is <%d>.",
                   type.code, type.bits);
    default:
      PADDLE_THROW("Unsupport type.bits %d", type.bits);
  }
}

void TensorFromDLPack(const ::DLTensor& dl_tensor, framework::Tensor* dst) {
  platform::CPUPlace dst_place = platform::CPUPlace();
  platform::CPUPlace src_place = platform::CPUPlace();

  std::vector<int64_t> vec;
  std::copy(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim,
            std::back_inserter(vec));

  framework::DDim vddim = framework::make_ddim(vec);

  dst->Resize(vddim);
  ::DLDataType type = dl_tensor.dtype;
  void* dst_ptr = GetDstPtrByDLDataType(type, dst, dst_place);

  auto src_ptr = static_cast<const void*>(dl_tensor.data);
  auto size = paddle::framework::product(vddim) * type.bits / 8;

  if (dl_tensor.ctx.device_type == kDLCPU) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  if (dl_tensor.ctx.device_type == kDLGPU) {
    platform::CUDAPlace dst_place =
        platform::CUDAPlace(dl_tensor.ctx.device_id);
    platform::CUDAPlace src_place =
        platform::CUDAPlace(dl_tensor.ctx.device_id);
    dst_ptr = GetDstPtrByDLDataType(type, dst, dst_place);
    auto* ctx = platform::DeviceContextPool::Instance().GetByPlace(dst_place);
    memory::Copy(
        dst_place, dst_ptr, src_place, src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(*ctx).stream());
  }
#endif
}

template <typename T>
std::ostream& print_tensor(std::ostream& os, const framework::Tensor& tensor) {
  auto inspect = tensor.data<T>();
  auto element_num = tensor.numel();

  os << "  - data: [";
  if (element_num > 0) {
    os << inspect[0];
    for (int j = 1; j < element_num; ++j) {
      os << " " << inspect[j];
    }
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
  os << "  - place: " << t.place() << "\n";
  os << "  - shape: [" << t.dims() << "]\n";
  os << "  - layout: " << DataLayoutToString(t.layout()) << "\n";

  Tensor tensor;
  tensor.Resize(t.dims());
  if (platform::is_cpu_place(t.place())) {
    tensor.ShareDataWith(t);
  } else {
    platform::CPUPlace place;
    framework::TensorCopy(t, place, &tensor);
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(t.place());
    dev_ctx.Wait();
  }

#define PrintTensorCallback(cpp_type, proto_type) \
  do {                                            \
    if (tensor.type() == proto_type) {            \
      os << "  - dtype: " << proto_type << "\n";  \
      print_tensor<cpp_type>(os, tensor);         \
      return os;                                  \
    }                                             \
  } while (0)

  _ForEachDataType_(PrintTensorCallback);
  VLOG(1) << "PrintVar: unrecognized data type:" << t.type();
  return os;
}

}  // namespace framework
}  // namespace paddle
