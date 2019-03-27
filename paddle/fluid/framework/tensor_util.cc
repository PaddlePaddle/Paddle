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
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/cryption.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {

void TensorCopy(const Tensor& src, const platform::Place& dst_place,
                const platform::DeviceContext& ctx, Tensor* dst) {
  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;
  src.check_memory_size();

  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  auto src_place = src.place();
  auto src_ptr = src.data<void>();

  auto dst_ptr = dst->mutable_data(dst_place, src.type());

  auto size = src.numel() * SizeOfType(src.type());

  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
#ifdef PADDLE_WITH_MKLDNN
    if (src.layout() == DataLayout::kMKLDNN) {
      dst->set_mkldnn_prim_desc(src.get_mkldnn_prim_desc());
    }
#endif
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
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = boost::get<platform::CPUPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::CUDAPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::CUDAPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(dst_gpu_place, ctx_gpu_place);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    memory::Copy(dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);
  } else if (platform::is_gpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = boost::get<platform::CUDAPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::CUDAPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream();
    if (platform::is_same_place(src_place, dst_place)) {
      if (src_ptr == dst_ptr) {
        VLOG(3) << "Skip copy the same data async from " << src_place << " to "
                << dst_place;
        return;
      }
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
  VLOG(3) << "TensorCopySync " << src.dims() << " from " << src.place()
          << " to " << dst_place;
  src.check_memory_size();
  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  auto src_place = src.place();
  auto src_ptr = src.data<void>();
  auto dst_ptr = dst->mutable_data(dst_place, src.type());
  auto size = src.numel() * SizeOfType(src.type());
  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr,
                 boost::get<platform::CPUPlace>(src_place), src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    platform::RecordEvent record_event("TensorCopy:GPU->CPU");
    auto src_gpu_place = boost::get<platform::CUDAPlace>(src_place);
    auto dst_cpu_place = boost::get<platform::CPUPlace>(dst_place);
    memory::Copy(dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, nullptr);
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    platform::RecordEvent record_event("TensorCopy:CPU->GPU");
    auto src_cpu_place = boost::get<platform::CPUPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::CUDAPlace>(dst_place);
    memory::Copy(dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, nullptr);
  } else if (platform::is_gpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    platform::RecordEvent record_event("TensorCopy:GPU->GPU");
    if (src_ptr == dst_ptr && platform::is_same_place(src_place, dst_place)) {
      VLOG(3) << "Skip copy the same data from " << src_place << " to "
              << dst_place;
      return;
    }
    auto src_gpu_place = boost::get<platform::CUDAPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::CUDAPlace>(dst_place);
    memory::Copy(dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, nullptr);
  } else if (platform::is_cuda_pinned_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    platform::RecordEvent record_event("TensorCopy:CUDAPinned->GPU");
    auto src_pinned_place = boost::get<platform::CUDAPinnedPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::CUDAPlace>(dst_place);
    memory::Copy(dst_gpu_place, dst_ptr, src_pinned_place, src_ptr, size,
                 nullptr);
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
                    const platform::DeviceContext& dev_ctx,
                    const bool encrypt) {
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
    PADDLE_ENFORCE(size < std::numeric_limits<std::streamsize>::max(),
                   "Index overflow when writing tensor");
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
                     boost::get<platform::CUDAPlace>(tensor.place()),
                     reinterpret_cast<const void*>(data), size_to_write,
                     gpu_dev_ctx.stream());
        gpu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW("Unexpected branch");
#endif
    } else {
      if (encrypt == true) {
        framework::Cryption& cryptor =
            *framework::Cryption::GetCryptorInstance();
        size_t encrypt_len = (static_cast<size_t>(size) / 16) * 16;
        size_t tail_len = static_cast<size_t>(size) - encrypt_len;
        std::string encrypt_data = cryptor.EncryptInMemory(
            static_cast<const char*>(data_ptr), encrypt_len);
        os.write(encrypt_data.data(), encrypt_len);
        if (tail_len > 0) {
          os.write(static_cast<const char*>(data_ptr) + encrypt_len, tail_len);
        }
      } else {
        os.write(static_cast<const char*>(data_ptr),
                 static_cast<std::streamsize>(size));
      }
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
                      const bool decrypt) {
  uint32_t version;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));
  PADDLE_ENFORCE_EQ(version, 0U, "Only version 0 is supported");
  proto::VarType::TensorDesc desc;
  {  // int32_t size
     // proto buffer
    int32_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char*>(buf.get()), size);
    PADDLE_ENFORCE(desc.ParseFromArray(buf.get(), size),
                   "Cannot parse tensor desc");
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
      PADDLE_THROW("Unexpected branch");
#endif
    } else {
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);
      if (decrypt == true) {
        framework::Cryption& cryptor =
            *framework::Cryption::GetCryptorInstance();
        size_t decrypt_len = (size / 16) * 16;
        std::string decrypt_data =
            cryptor.DecryptInMemory(static_cast<const char*>(buf), decrypt_len);
        platform::CPUPlace cpu;
        memory::Copy(cpu, buf, cpu, decrypt_data.data(), decrypt_len);
      }
    }
  }
}

}  // namespace framework
}  // namespace paddle
