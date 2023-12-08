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

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

#if defined(__CUDACC__) && CUDA_VERSION >= 11000
#define CUSTOMAR_ENABLE_BF16
#endif

namespace paddle {
namespace operators {

constexpr int DEFAULT_BLOCK_SIZE = 1024;
constexpr int MAX_ALL_REDUCE_BLOCKS = 24;

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void st_flag_release(uint32_t &flag,  // NOLINT
                                              volatile uint32_t *flag_addr) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
#else
  __threadfence_system();
  asm volatile("st.global.volatile.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void ld_flag_acquire(uint32_t &flag,  // NOLINT
                                              volatile uint32_t *flag_addr) {
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
#else
  asm volatile("ld.global.volatile.b32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

class SystemCUDAAllocator : public phi::Allocator {
 public:
  static phi::Allocator *Instance() {
    static SystemCUDAAllocator allocator;
    return &allocator;
  }

  phi::Allocator::AllocationPtr Allocate(size_t size) override {
    if (size == 0) {
      return nullptr;
    }
    void *ptr = nullptr;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&ptr, size));
    return phi::Allocator::AllocationPtr(new phi::Allocation(ptr, size, place),
                                         DeleteFunc);
  }

  bool IsAllocThreadSafe() const override { return true; }

 private:
  static void DeleteFunc(phi::Allocation *allocation) {
    cudaFree(allocation->ptr());
    delete allocation;
  }

  SystemCUDAAllocator() : place(platform::GetCurrentDeviceId()) {}

  DISABLE_COPY_AND_ASSIGN(SystemCUDAAllocator);

 private:
  phi::GPUPlace place;
};

template <typename T>
static __global__ void FillBarrierValue(T *x, T value) {
  x[threadIdx.x] = value;
}

template <typename T, int N>
static __forceinline__ __device__ void BarrierAllGPUs(
    const phi::Array<volatile T *, N> &barriers, T barrier_value, int rank) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;

  if (thread_id < N) {
    if (block_id == 0) {
      barriers[thread_id][rank] = barrier_value;
    }
    while (barriers[rank][thread_id] < barrier_value) {
    }
  }

  __syncthreads();
}

template <typename T, int N>
static __forceinline__ __device__ void BarrierAllGPUsAllBlock(
    const phi::Array<volatile T *, N> &barriers, T barrier_value, int rank) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;

  if (thread_id < N) {
    uint32_t flag_block_offset = N + block_id * N;
    st_flag_release(barrier_value,
                    barriers[thread_id] + flag_block_offset + rank);
    uint32_t rank_barrier = 0;
    volatile uint32_t *peer_barrier_d =
        barriers[rank] + flag_block_offset + thread_id;
    do {
      ld_flag_acquire(rank_barrier, peer_barrier_d);
    } while (rank_barrier != barrier_value);
  }

  __syncthreads();
}

template <typename T, int N>
struct AlignedVectorAddHelper {
  DEVICE static void Run(const phi::AlignedVector<T, N> &in,
                         phi::AlignedVector<T, N> *out) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      (*out)[i] += in[i];
    }
  }
};

template <int N>
struct AlignedVectorAddHelper<phi::dtype::float16, N> {
  DEVICE static void Run(const phi::AlignedVector<phi::dtype::float16, N> &in,
                         phi::AlignedVector<phi::dtype::float16, N> *out) {
    const __half2 *in_ptr =
        static_cast<const __half2 *>(static_cast<const void *>(&in[0]));
    __half2 *out_ptr = static_cast<__half2 *>(static_cast<void *>(&(*out)[0]));
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      out_ptr[i] = __hadd2(out_ptr[i], in_ptr[i]);
    }
    if (N % 2 != 0) {
      (*out)[N - 1] += in[N - 1];
    }
  }
};
#ifdef CUSTOMAR_ENABLE_BF16
inline __device__ __nv_bfloat162 float2bf162(const float2 a) {
  __nv_bfloat162 a_;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  a_ = __float22bfloat162_rn(a);
#else
  a_.x = __float2bfloat16_rn(a.x);
  a_.y = __float2bfloat16_rn(a.y);
#endif
  return a_;
}

// Bfloat16 Specialization.
template <int N>
struct AlignedVectorAddHelper<phi::dtype::bfloat16, N> {
  DEVICE static void Run(const phi::AlignedVector<phi::dtype::bfloat16, N> &in,
                         phi::AlignedVector<phi::dtype::bfloat16, N> *out) {
    const __nv_bfloat162 *in_ptr =
        static_cast<const __nv_bfloat162 *>(static_cast<const void *>(&in[0]));
    __nv_bfloat162 *out_ptr =
        static_cast<__nv_bfloat162 *>(static_cast<void *>(&(*out)[0]));
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
      out_ptr[i] = __hadd2(out_ptr[i], in_ptr[i]);
#else
      float2 out{};
      out.x = __bfloat162float(out_ptr[i].x) + __bfloat162float(in_ptr[i].x);
      out.y = __bfloat162float(out_ptr[i].y) + __bfloat162float(in_ptr[i].y);
      out_ptr[i] = float2bf162(out);
#endif
    }
    if (N % 2 != 0) {
      (*out)[N - 1] += in[N - 1];
    }
  }
};

#endif //CUSTOMAR_ENABLE_BF16

template <typename T, int N, int VecSize, bool HasLeftValue = true>
static __device__ __forceinline__ void AllReduceFunc(
    const phi::Array<T *, N> &ins,
    int idx,
    int stride,
    int n,
    int rank,
    T *out) {
  using AlignedVec = phi::AlignedVector<T, VecSize>;
  while (idx + VecSize <= n) {
    AlignedVec in_vecs[N];

#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto cur_rank = (i + rank) % N;
      const auto *ptr = ins[cur_rank] + idx;
      phi::Load(ptr, &in_vecs[cur_rank]);
    }

#pragma unroll
    for (int i = 1; i < N; ++i) {
      AlignedVectorAddHelper<T, VecSize>::Run(in_vecs[i], &in_vecs[0]);
    }
    phi::Store(in_vecs[0], out + idx);
    idx += stride;
  }

  while (HasLeftValue && idx < n) {
    T sum = ins[0][idx];
#pragma unroll
    for (int i = 1; i < N; ++i) {
      sum += ins[i][idx];
    }
    out[idx] = sum;
    ++idx;
  }
}

template <typename T, typename BarrierT, int N, int VecSize>
static __global__
__launch_bounds__(DEFAULT_BLOCK_SIZE) void OneShotAllReduceKernel(
    phi::Array<T *, N> ins,
    phi::Array<volatile BarrierT *, N> barriers,
    BarrierT barrier_value,
    int rank,
    size_t n,
    T *out) {
  BarrierAllGPUs<BarrierT, N>(barriers, barrier_value, rank);

  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  int stride = (blockDim.x * gridDim.x) * VecSize;
  AllReduceFunc<T, N, VecSize>(ins, idx, stride, n, rank, out);
}

template <typename T, int VecSize>
static __device__ __forceinline__ void VecStoreGlobalMem(const T *x, T *y) {
  using AlignedVec = phi::AlignedVector<T, VecSize>;
  const auto *x_vec =
      static_cast<const AlignedVec *>(static_cast<const void *>(x));
  auto *y_vec = static_cast<AlignedVec *>(static_cast<void *>(y));
  y_vec[0] = x_vec[0];
}

template <typename T, typename BarrierT, int N, int VecSize>
static __global__
__launch_bounds__(DEFAULT_BLOCK_SIZE) void TwoShotAllReduceKernel(
    phi::Array<T *, N> ins,
    phi::Array<volatile BarrierT *, N> barriers,
    BarrierT barrier_value,
    int rank,
    size_t n,
    T *out) {
  BarrierAllGPUs<BarrierT, N>(barriers, barrier_value, rank);
  const size_t n_per_gpu = n / N;
  int idx =
      (threadIdx.x + blockIdx.x * blockDim.x) * VecSize + rank * n_per_gpu;
  int stride = (blockDim.x * gridDim.x) * VecSize;
  int limit = (rank + 1) * n_per_gpu;
  AllReduceFunc<T, N, VecSize, false>(ins, idx, stride, limit, rank, ins[rank]);

  BarrierAllGPUsAllBlock<BarrierT, N>(barriers, barrier_value + 1, rank);
  using AlignedVec = phi::AlignedVector<T, VecSize>;

  int dst_offset[N];
  int dst_rank[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    int tmp = (i + rank) % N;
    dst_rank[i] = tmp;
    dst_offset[i] = (tmp - rank) * n_per_gpu;
  }

  while (idx + VecSize <= limit) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto dst_idx = idx + dst_offset[i];
      VecStoreGlobalMem<T, VecSize>(ins[dst_rank[i]] + dst_idx, out + dst_idx);
    }
    idx += stride;
  }
}

class CustomNCCLComm {
 public:
  virtual void SwapInput(phi::DenseTensor *x) = 0;
  virtual phi::DenseTensor AllReduce() = 0;

  virtual ~CustomNCCLComm() = default;

 protected:
  void EnableP2P(int nranks) {
    for (int i = 0; i < nranks; ++i) {
      platform::CUDADeviceGuard guard(i);
      for (int j = 0; j < nranks; ++j) {
        int enabled = 0;
        PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceCanAccessPeer(&enabled, i, j));
        PADDLE_ENFORCE_EQ(enabled, 1);
        PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceEnablePeerAccess(j, 0));
      }
    }
  }
};

template <int N>
class CustomNCCLCommImpl : public CustomNCCLComm {
 private:
  template <typename T>
  struct P2PBuffer {
    template <typename InitFunc>
    P2PBuffer(CustomNCCLCommImpl<N> *comm, size_t size, InitFunc &&init_func) {
      phi::Dim<1> dim;
      dim[0] = static_cast<int64_t>(size);
      t_.Resize(dim);
      void *ptr = t_.AllocateFrom(SystemCUDAAllocator::Instance(),
                                  phi::CppTypeToDataType<T>::Type());
      init_func(*(comm->ctx_), &t_);
      comm->ctx_->Wait();

      comm->Barrier();

      auto pids = comm->AllGatherPOD(::getpid());
      for (int i = 0; i < N; ++i) {
        BroadcastDevicePtr(comm, ptr, i, pids[0]);
      }
    }

    ~P2PBuffer() {
      for (int i = 0; i < N; ++i) {
        if (i != rank_) {
          cudaIpcCloseMemHandle(ptrs_[i]);
        }
        ::munmap(mmap_ptrs_[i], sizeof(cudaIpcMemHandle_t));
        ::shm_unlink(shm_names_[i].c_str());
      }
      t_.clear();
    }

    const phi::DenseTensor &GetTensor() const { return t_; }
    phi::DenseTensor *GetMutableTensor() { return &t_; }

    template <typename U = T>
    phi::Array<U *, N> GetPtrs() const {
      phi::Array<U *, N> results;
#pragma unroll
      for (int i = 0; i < N; ++i) {
        results[i] = static_cast<U *>(ptrs_[i]);
      }
      return results;
    }

   private:
    void BroadcastDevicePtr(CustomNCCLCommImpl<N> *comm,
                            void *ptr,
                            int cur_rank,
                            pid_t pid) {
      VLOG(10) << "BroadcastDevicePtr starts " << cur_rank << " -> "
               << comm->rank_;
      std::string name = "/paddle_custom_nccl_" + std::to_string(pid) + "_" +
                         std::to_string(cur_rank);
      cudaIpcMemHandle_t *handle;
      bool is_root = (comm->rank_ == cur_rank);

      if (!is_root) {
        comm->Barrier();
      }

      int fd = ::shm_open(
          name.c_str(), is_root ? (O_RDWR | O_CREAT) : O_RDONLY, 0600);
      PADDLE_ENFORCE_GE(fd, 0);
      if (is_root) {
        PADDLE_ENFORCE_EQ(ftruncate(fd, sizeof(cudaIpcMemHandle_t)), 0);
      }
      void *mmap_ptr = ::mmap(nullptr,
                              sizeof(cudaIpcMemHandle_t),
                              is_root ? (PROT_READ | PROT_WRITE) : PROT_READ,
                              MAP_SHARED,
                              fd,
                              0);
      PADDLE_ENFORCE_NOT_NULL(mmap_ptr);
      PADDLE_ENFORCE_NE(mmap_ptr, MAP_FAILED);
      handle = static_cast<cudaIpcMemHandle_t *>(mmap_ptr);
      if (is_root) {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcGetMemHandle(handle, ptr));
        ptrs_[cur_rank] = ptr;
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcOpenMemHandle(
            &ptrs_[cur_rank], *handle, cudaIpcMemLazyEnablePeerAccess));
      }
      if (is_root) {
        comm->Barrier();
      }

      comm->Barrier();
      mmap_ptrs_[cur_rank] = mmap_ptr;
      shm_names_[cur_rank] = name;
      VLOG(10) << "BroadcastDevicePtr ends " << cur_rank << " -> "
               << comm->rank_;
    }

   private:
    phi::Array<void *, N> ptrs_;
    phi::DenseTensor t_;
    int rank_;
    phi::Array<void *, N> mmap_ptrs_;
    phi::Array<std::string, N> shm_names_;
  };

 public:
  using BarrierDType = uint32_t;
  using BarrierTensorDType = int32_t;

  static_assert(sizeof(BarrierDType) == sizeof(BarrierTensorDType),
                "Size not match");

  CustomNCCLCommImpl(const phi::GPUContext &ctx,
                     size_t one_shot_max_size,
                     size_t two_shot_max_size,
                     int ring_id)
      : ctx_(&ctx),
        one_shot_max_size_(one_shot_max_size),
        two_shot_max_size_(two_shot_max_size) {
    PADDLE_ENFORCE_LT(one_shot_max_size, two_shot_max_size);
    auto comm =
        platform::NCCLCommContext::Instance().Get(ring_id, ctx.GetPlace());
    comm_ = comm->comm();
    rank_ = comm->rank();
    auto nranks = comm->nranks();
    PADDLE_ENFORCE_EQ(
        nranks,
        N,
        phi::errors::InvalidArgument("Invalid world size, this may be a bug."));

    barrier_value_ = 0;
    VLOG(10) << "CustomNCCLCommImpl::CustomNCCLCommImpl";
    ins_ = std::make_unique<P2PBuffer<uint8_t>>(
        this,
        two_shot_max_size_,
        [](const phi::GPUContext &ctx, phi::DenseTensor *t) {});
    VLOG(10) << "CustomNCCLCommImpl::ins_ inited";

    barriers_ = std::make_unique<P2PBuffer<BarrierTensorDType>>(
        this,
        N * (MAX_ALL_REDUCE_BLOCKS + 1),
        [](const phi::GPUContext &ctx, phi::DenseTensor *t) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              cudaMemsetAsync(t->data(),
                              0,
                              t->numel() * sizeof(BarrierTensorDType),
                              ctx.stream()));
        });
    VLOG(10) << "CustomNCCLCommImpl::barriers_ inited";
  }

  void SwapInput(phi::DenseTensor *x) override {
    out_ = *x;
    auto numel = x->numel();
    auto dtype = x->dtype();
    auto algo = ChooseAlgo(numel, dtype);
    if (algo <= 2 && !HasReachMaxBarrierValue(algo)) {
      ShareTensor(x, ins_->GetMutableTensor());
    }
  }

  phi::DenseTensor AllReduce() override {
    auto dtype = out_.dtype();
    auto numel = out_.numel();
    auto algo = ChooseAlgo(numel, dtype);
    if (algo > 2) {
      NCCLAllReduce(out_.data(), numel, dtype);
      return std::move(out_);
    }

    if (HasReachMaxBarrierValue(algo)) {
      NCCLAllReduce(out_.data(), numel, dtype);
      ResetBarriers();
      return std::move(out_);
    }

#define PD_CUSTOM_ALLREDUCE(__cpp_dtype, __vec_size)                 \
  do {                                                               \
    if (dtype == ::phi::CppTypeToDataType<__cpp_dtype>::Type()) {    \
      if (algo == 1) {                                               \
        return OneShotAllReduceImpl<__cpp_dtype, __vec_size>(numel); \
      } else {                                                       \
        return TwoShotAllReduceImpl<__cpp_dtype, __vec_size>(numel); \
      }                                                              \
    }                                                                \
  } while (0)
    PD_CUSTOM_ALLREDUCE(phi::dtype::bfloat16, 8);
    PD_CUSTOM_ALLREDUCE(phi::dtype::float16, 8);
    PD_CUSTOM_ALLREDUCE(float, 4);
    PD_CUSTOM_ALLREDUCE(double, 2);
    PADDLE_THROW(
        phi::errors::InvalidArgument("Unsupported data type %s", dtype));
  }

 private:
  uint32_t ChooseAlgo(size_t numel, phi::DataType dtype) const {
    auto sizeof_dtype = phi::SizeOf(dtype);
    auto mem_size = numel * sizeof_dtype;
    if (mem_size <= one_shot_max_size_) {
      return 1;
    } else if (mem_size <= two_shot_max_size_ && numel % N == 0 &&
               (numel / N) % (16 / sizeof_dtype) == 0) {
      return 2;
    } else {
      return 3;
    }
  }

  void NCCLAllReduce(void *ptr, size_t numel, phi::DataType dtype) {
    auto nccl_dtype = platform::ToNCCLDataType(dtype);
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
        ptr, ptr, numel, nccl_dtype, ncclSum, comm_, ctx_->stream()));
  }

  void ResetBarriers() {
    LOG(INFO) << "barrier_value_ " << barrier_value_ << " , restart barrier";
    Barrier();
    auto *barrier_tensor = barriers_->GetMutableTensor();
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemsetAsync(barrier_tensor->data(),
                        0,
                        barrier_tensor->numel() * sizeof(BarrierTensorDType),
                        ctx_->stream()));
    Barrier();
    barrier_value_ = 0;
  }

  bool HasReachMaxBarrierValue(int algo) const {
    return barrier_value_ > std::numeric_limits<BarrierDType>::max() - algo;
  }

  template <typename T, int VecSize>
  phi::DenseTensor OneShotAllReduceImpl(int64_t numel) {
    const auto &in_ptrs = ins_->template GetPtrs<T>();
    const auto &barrier_ptrs =
        barriers_->template GetPtrs<volatile BarrierDType>();
    auto *out_data = out_.template data<T>();
    ++barrier_value_;

    int threads = DEFAULT_BLOCK_SIZE;  // 1024
    PADDLE_ENFORCE_GE(threads, N);
    int64_t blocks = ((numel + VecSize - 1) / VecSize + threads - 1) / threads;
    blocks = std::min<int64_t>(blocks, MAX_ALL_REDUCE_BLOCKS /*24*/);
    VLOG(10) << "Use OneShotAllReduceKernel for size = " << numel;

    OneShotAllReduceKernel<T, BarrierDType, N, VecSize>
        <<<blocks, threads, 0, ctx_->stream()>>>(
            in_ptrs, barrier_ptrs, barrier_value_, rank_, numel, out_data);
    return std::move(out_);
  }

  template <typename T, int VecSize>
  phi::DenseTensor TwoShotAllReduceImpl(int64_t numel) {
    PADDLE_ENFORCE_EQ(numel % N, 0);
    const auto &in_ptrs = ins_->template GetPtrs<T>();
    const auto &barrier_ptrs =
        barriers_->template GetPtrs<volatile BarrierDType>();
    auto *out_data = out_.template data<T>();
    if (barrier_value_ > 0) {
      barrier_value_ += 2;
    } else {
      barrier_value_ = 1;
    }

    // int threads = ctx_->GetMaxThreadsPerBlock();
    int threads = DEFAULT_BLOCK_SIZE;
    PADDLE_ENFORCE_GE(threads, N);
    int32_t blocks =
        ((numel / N + VecSize - 1) / VecSize + threads - 1) / threads;
    blocks = std::min<int64_t>(blocks, MAX_ALL_REDUCE_BLOCKS /*24*/);
    VLOG(10) << "Use TwoShotAllReduceKernel for size = " << numel;
    TwoShotAllReduceKernel<T, BarrierDType, N, VecSize>
        <<<blocks, threads, 0, ctx_->stream()>>>(
            in_ptrs, barrier_ptrs, barrier_value_, rank_, numel, out_data);
    return std::move(out_);
  }

  void ShareTensor(phi::DenseTensor *x, phi::DenseTensor *y) {
    PADDLE_ENFORCE_LE(x->numel(), two_shot_max_size_);
    const void *y_ptr = y->data();
    y->Resize(x->dims());
    auto *new_y_ptr = ctx_->Alloc(y, x->dtype());
    PADDLE_ENFORCE_EQ(y_ptr, new_y_ptr);
    x->ShareBufferWith(*y);
  }

  void Barrier() { AllGatherPOD(1); }

  template <typename T>
  std::vector<T> AllGatherPOD(const T &value) {
    std::vector<T> result(N);
    AllGatherBuffer(&value, result.data(), sizeof(T));
    return result;
  }

  void AllGatherBuffer(const void *src, void *dst, size_t nbytes) {
    phi::DenseTensor tensor;
    phi::Dim<1> dim;
    dim[0] = N * nbytes;
    tensor.Resize(dim);
    auto *ptr = ctx_->template Alloc<uint8_t>(&tensor);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(ptr + rank_ * nbytes,
                                               src,
                                               nbytes,
                                               cudaMemcpyHostToDevice,
                                               ctx_->stream()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllGather(
        ptr + rank_ * nbytes, ptr, nbytes, ncclInt8, comm_, ctx_->stream()));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(
        dst, ptr, N * nbytes, cudaMemcpyDeviceToHost, ctx_->stream()));
    ctx_->Wait();
  }

 private:
  std::unique_ptr<P2PBuffer<uint8_t>> ins_;
  std::unique_ptr<P2PBuffer<BarrierTensorDType>> barriers_;
  BarrierDType barrier_value_;
  phi::DenseTensor out_;

  const phi::GPUContext *ctx_;
  size_t one_shot_max_size_;
  size_t two_shot_max_size_;
  ncclComm_t comm_;
  int rank_;
};

static std::unique_ptr<CustomNCCLComm> CreateCustomNCCLComm(
    const phi::GPUContext &ctx,
    int64_t one_shot_max_size,
    int64_t two_shot_max_size,
    int ring_id) {
  if (one_shot_max_size <= 0 || two_shot_max_size <= 0 ||
      one_shot_max_size >= two_shot_max_size) {
    return nullptr;
  }

  auto nranks = platform::NCCLCommContext::Instance()
                    .Get(ring_id, ctx.GetPlace())
                    ->nranks();
#define PD_CREATE_CUSTOM_NCCL_COMM(__nranks)                   \
  do {                                                         \
    if (nranks == __nranks) {                                  \
      return std::make_unique<CustomNCCLCommImpl<__nranks>>(   \
          ctx, one_shot_max_size, two_shot_max_size, ring_id); \
    }                                                          \
  } while (0)

  PD_CREATE_CUSTOM_NCCL_COMM(8);
  PD_CREATE_CUSTOM_NCCL_COMM(4);
  PD_CREATE_CUSTOM_NCCL_COMM(2);
  return nullptr;
}

}  // namespace operators
}  // namespace paddle
