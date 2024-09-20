// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"

namespace phi {
namespace fusion {

#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// join grad
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelWithCVM(const size_t N,
                                                 T **out_grads_values,
                                                 T **in_grads_values,
                                                 T **cvm_values,
                                                 size_t **lods_values,
                                                 const int batch_size,
                                                 const int embedding_size,
                                                 const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    T &val = (offset < cvm_offset)
                 ? *(cvm_values[x] + y * cvm_offset + offset)
                 : *(out_grads_values[x] + y * embedding_size + offset);

    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}

// join only show not has click
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelWithShow(const size_t N,
                                                  T **out_grads_values,
                                                  T **in_grads_values,
                                                  T **cvm_values,
                                                  size_t **lods_values,
                                                  const int batch_size,
                                                  const int embedding_size,
                                                  const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    T &val =
        (offset < cvm_offset)
            ? *(cvm_values[x] + y * cvm_offset + offset)
            : *(out_grads_values[x] + y * (embedding_size - 1) + offset - 1);

    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}

// update grad
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelNoCVM(const size_t N,
                                               T **out_grads_values,
                                               T **in_grads_values,
                                               T **cvm_values,
                                               size_t **lods_values,
                                               const int batch_size,
                                               const int embedding_size,
                                               const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    T &val = (offset < cvm_offset)
                 ? *(cvm_values[x] + y * cvm_offset + offset)
                 : *(out_grads_values[x] + y * (embedding_size - cvm_offset) +
                     offset - cvm_offset);

    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}

template <typename T>
void FusedSeqpoolCVMGrad(const phi::GPUContext &dev_ctx,
                         const std::vector<const T *> &out_grads_data,
                         const std::vector<T *> &in_grads_data,
                         const std::vector<const T *> &cvm_data,
                         const std::vector<const size_t *> &lods,
                         const int batch_size,
                         const int slot_num,
                         const int embedding_size,
                         const bool use_cvm,
                         const int cvm_offset) {
  auto stream = dev_ctx.stream();
  size_t total_ptr_len = out_grads_data.size() + in_grads_data.size() +
                         cvm_data.size() + lods.size();
  auto temp_ptr = phi::memory_utils::AllocShared(
      dev_ctx.GetPlace(), total_ptr_len * sizeof(void *));
#ifdef PADDLE_WITH_HIP
  T **gpu_out_grads_values = reinterpret_cast<T **>(temp_ptr->ptr());
  phi::backends::gpu::GpuMemcpyAsync(gpu_out_grads_values,
                                     out_grads_data.data(),
                                     out_grads_data.size() * sizeof(T *),
                                     hipMemcpyHostToDevice,
                                     stream);

  T **gpu_in_grads_values =
      reinterpret_cast<T **>(&gpu_out_grads_values[out_grads_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(gpu_in_grads_values,
                                     in_grads_data.data(),
                                     in_grads_data.size() * sizeof(T *),
                                     hipMemcpyHostToDevice,
                                     stream);

  T **gpu_cvm_values =
      reinterpret_cast<T **>(&gpu_in_grads_values[in_grads_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(gpu_cvm_values,
                                     cvm_data.data(),
                                     cvm_data.size() * sizeof(T *),
                                     hipMemcpyHostToDevice,
                                     stream);

  size_t **lods_values =
      reinterpret_cast<size_t **>(&gpu_cvm_values[cvm_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(lods_values,
                                     lods.data(),
                                     lods.size() * sizeof(size_t *),
                                     hipMemcpyHostToDevice,
                                     stream);
#else
  T **gpu_out_grads_values = reinterpret_cast<T **>(temp_ptr->ptr());
  phi::backends::gpu::GpuMemcpyAsync(gpu_out_grads_values,
                                     out_grads_data.data(),
                                     out_grads_data.size() * sizeof(T *),
                                     cudaMemcpyHostToDevice,
                                     stream);

  T **gpu_in_grads_values =
      reinterpret_cast<T **>(&gpu_out_grads_values[out_grads_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(gpu_in_grads_values,
                                     in_grads_data.data(),
                                     in_grads_data.size() * sizeof(T *),
                                     cudaMemcpyHostToDevice,
                                     stream);

  T **gpu_cvm_values =
      reinterpret_cast<T **>(&gpu_in_grads_values[in_grads_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(gpu_cvm_values,
                                     cvm_data.data(),
                                     cvm_data.size() * sizeof(T *),
                                     cudaMemcpyHostToDevice,
                                     stream);

  size_t **lods_values =
      reinterpret_cast<size_t **>(&gpu_cvm_values[cvm_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(lods_values,
                                     lods.data(),
                                     lods.size() * sizeof(size_t *),
                                     cudaMemcpyHostToDevice,
                                     stream);
#endif

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, N);
  if (use_cvm) {
    // join grad
    FusedSeqpoolCVMGradKernelWithCVM<<<config.block_per_grid.x,
                                       config.thread_per_block.x,
                                       0,
                                       stream>>>(N,
                                                 gpu_out_grads_values,
                                                 gpu_in_grads_values,
                                                 gpu_cvm_values,
                                                 lods_values,
                                                 batch_size,
                                                 embedding_size,
                                                 cvm_offset);
  } else {
    // update grad
    FusedSeqpoolCVMGradKernelNoCVM<<<config.block_per_grid.x,
                                     config.thread_per_block.x,
                                     0,
                                     stream>>>(N,
                                               gpu_out_grads_values,
                                               gpu_in_grads_values,
                                               gpu_cvm_values,
                                               lods_values,
                                               batch_size,
                                               embedding_size,
                                               cvm_offset);
  }
}

template <typename T, typename Context>
void FusedSeqpoolCVMGradCUDAKernel(
    const Context &dev_ctx,
    const std::vector<const DenseTensor *> &x,
    const DenseTensor &cvm_in,
    const std::vector<const DenseTensor *> &out_grad,
    const std::string &pooltype,
    float pad_value,
    bool use_cvm,
    int cvm_offset,
    std::vector<DenseTensor *> x_grad,
    DenseTensor *cvm_grad) {
  auto &out_grads = out_grad;
  auto &in_grads = x_grad;
  auto *cvm = &cvm_in;

  const auto slot_size = in_grads.size();
  std::vector<const T *> out_grads_data(slot_size);
  std::vector<T *> in_grads_data(slot_size);
  std::vector<const T *> cvm_data(slot_size);
  std::vector<const size_t *> lods_data(slot_size);

  int embedding_size = in_grads[0]->numel() / in_grads[0]->dims()[0];
  int batch_size = -1;
  std::vector<phi::MixVector<size_t> *> mix_lods_v(slot_size);

  for (size_t i = 0; i < slot_size; ++i) {
    auto *in_grad = in_grads[i];

    Vector<size_t> lods;
    if (in_grad->lod().size() != 0) {
      auto lod = in_grad->lod();
      lods = lod[0];
    } else {
      lods.push_back(0);
      for (int i = 0; i < in_grad->dims()[0]; i++) {
        lods.push_back(i + 1);
      }
    }

    int cur_batch_size = in_grad->lod().size() ? in_grad->lod()[0].size() - 1
                                               : in_grad->dims()[0];
    if (batch_size == -1) {
      batch_size = cur_batch_size;
    } else {
      PADDLE_ENFORCE_EQ(batch_size,
                        cur_batch_size,
                        common::errors::PreconditionNotMet(
                            "The batch size of all input should be same, "
                            "please cheack, last batchsize is %d, current "
                            "batchsize is %d",
                            batch_size,
                            cur_batch_size));
    }

    auto *out_grad = out_grads[i];
    out_grads_data[i] = reinterpret_cast<const T *>(out_grad->data<T>());

    in_grads_data[i] = reinterpret_cast<T *>(
        dev_ctx.template Alloc<T>(in_grad, in_grad->numel() * sizeof(T)));
    mix_lods_v[i] = new phi::MixVector<size_t>(&lods);
    lods_data[i] = mix_lods_v[i]->CUDAData(dev_ctx.GetPlace());
    cvm_data[i] = reinterpret_cast<const T *>(cvm->data<T>());
  }
  FusedSeqpoolCVMGrad(dev_ctx,
                      out_grads_data,
                      in_grads_data,
                      cvm_data,
                      lods_data,
                      batch_size,
                      slot_size,
                      embedding_size,
                      use_cvm,
                      cvm_offset);

  for (int i = 0; i < slot_size; i++) {
    delete mix_lods_v[i];
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_seqpool_cvm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedSeqpoolCVMGradCUDAKernel,
                   float) {}
