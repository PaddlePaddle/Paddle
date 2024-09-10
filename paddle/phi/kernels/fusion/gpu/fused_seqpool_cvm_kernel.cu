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

// normal
template <typename T>
__global__ void FusedSeqpoolKernelNormal(const size_t N,
                                         T **input_values,
                                         T **seqpool_output_values,
                                         size_t **lods_values,
                                         const int batch_size,
                                         const int embedding_size,
                                         const float pad_value) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    T val = static_cast<T>(pad_value);
    for (auto k = start; k < end; ++k) {
      val += *(input_values[x] + k * embedding_size + offset);
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}

// join need show click input
template <typename T>
__global__ void FusedCVMKernelWithCVM(const size_t N,
                                      T **output_values,
                                      T **seqpool_output_values,
                                      const int batch_size,
                                      const int embedding_size,
                                      const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    if (offset == 0) {         // show
      *(output_values[x] + y * embedding_size) =
          log(*(seqpool_output_values[x] + y * embedding_size) + 1);
    } else if (offset == 1) {  // click
      *(output_values[x] + y * embedding_size + offset) =
          log(*(seqpool_output_values[x] + y * embedding_size + 1) + 1) -
          log(*(seqpool_output_values[x] + y * embedding_size) + 1);
    } else {
      *(output_values[x] + y * embedding_size + offset) =
          *(seqpool_output_values[x] + y * embedding_size + offset);
    }
  }
}

// update not need show click input
template <typename T>
__global__ void FusedCVMKernelNoCVM(const size_t N,
                                    T **output_values,
                                    T **seqpool_output_values,
                                    const int batch_size,
                                    const int no_cvm_embedding_size,
                                    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / no_cvm_embedding_size;
    int offset = i % no_cvm_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    // no cvm
    *(output_values[x] + y * no_cvm_embedding_size + offset) =
        *(seqpool_output_values[x] + y * (no_cvm_embedding_size + cvm_offset) +
          offset + cvm_offset);
  }
}

template <typename T>
void FusedSeqpoolCVM(
    const phi::GPUContext &dev_ctx,  // const paddle::phi::Place &place,
    const std::vector<const T *> &input_data,
    const std::vector<T *> &output_data,
    const std::vector<T *> &seqpool_output_data,
    std::vector<const size_t *> lods,
    const int batch_size,
    const int slot_num,
    const int embedding_size,
    const float padding_value,
    const bool use_cvm,
    const int cvm_offset) {
  auto stream = dev_ctx.stream();
  size_t total_ptr_len = input_data.size() + output_data.size() +
                         seqpool_output_data.size() + lods.size();
  auto temp_ptr = phi::memory_utils::AllocShared(
      dev_ctx.GetPlace(), total_ptr_len * sizeof(void *));
  void *ptr = temp_ptr->ptr();

#ifdef PADDLE_WITH_HIP
  T **gpu_input_values = reinterpret_cast<T **>(temp_ptr->ptr());
  phi::backends::gpu::GpuMemcpyAsync(gpu_input_values,
                                     input_data.data(),
                                     input_data.size() * sizeof(T *),
                                     hipMemcpyHostToDevice,
                                     stream);
  T **gpu_output_values =
      reinterpret_cast<T **>(&gpu_input_values[input_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(gpu_output_values,
                                     output_data.data(),
                                     output_data.size() * sizeof(T *),
                                     hipMemcpyHostToDevice,
                                     stream);
  T **gpu_seqpool_output_values =
      reinterpret_cast<T **>(&gpu_output_values[output_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(gpu_seqpool_output_values,
                                     seqpool_output_data.data(),
                                     seqpool_output_data.size() * sizeof(T *),
                                     hipMemcpyHostToDevice,
                                     stream);
  size_t **lods_values = reinterpret_cast<size_t **>(
      &gpu_seqpool_output_values[seqpool_output_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(lods_values,
                                     lods.data(),
                                     lods.size() * sizeof(size_t *),
                                     hipMemcpyHostToDevice,
                                     stream);
#else
  T **gpu_input_values = reinterpret_cast<T **>(temp_ptr->ptr());
  phi::backends::gpu::GpuMemcpyAsync(gpu_input_values,
                                     input_data.data(),
                                     input_data.size() * sizeof(T *),
                                     cudaMemcpyHostToDevice,
                                     stream);
  T **gpu_output_values =
      reinterpret_cast<T **>(&gpu_input_values[input_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(gpu_output_values,
                                     output_data.data(),
                                     output_data.size() * sizeof(T *),
                                     cudaMemcpyHostToDevice,
                                     stream);
  T **gpu_seqpool_output_values =
      reinterpret_cast<T **>(&gpu_output_values[output_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(gpu_seqpool_output_values,
                                     seqpool_output_data.data(),
                                     seqpool_output_data.size() * sizeof(T *),
                                     cudaMemcpyHostToDevice,
                                     stream);
  size_t **lods_values = reinterpret_cast<size_t **>(
      &gpu_seqpool_output_values[seqpool_output_data.size()]);
  phi::backends::gpu::GpuMemcpyAsync(lods_values,
                                     lods.data(),
                                     lods.size() * sizeof(size_t *),
                                     cudaMemcpyHostToDevice,
                                     stream);
#endif

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  phi::backends::gpu::GpuLaunchConfig config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, N);
  // first sum pool
  FusedSeqpoolKernelNormal<<<config.block_per_grid.x,
                             config.thread_per_block.x,
                             0,
                             stream>>>(N,
                                       gpu_input_values,
                                       gpu_seqpool_output_values,
                                       lods_values,
                                       batch_size,
                                       embedding_size,
                                       padding_value);
  // second log
  if (use_cvm) {
    FusedCVMKernelWithCVM<<<config.block_per_grid.x,
                            config.thread_per_block.x,
                            0,
                            stream>>>(N,
                                      gpu_output_values,
                                      gpu_seqpool_output_values,
                                      batch_size,
                                      embedding_size,
                                      cvm_offset);
  } else {
    // not need show click input
    N = static_cast<size_t>(batch_size * slot_num *
                            (embedding_size - cvm_offset));
    phi::backends::gpu::GpuLaunchConfig config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, N);
    FusedCVMKernelNoCVM<<<config.block_per_grid.x,
                          config.thread_per_block.x,
                          0,
                          stream>>>(N,
                                    gpu_output_values,
                                    gpu_seqpool_output_values,
                                    batch_size,
                                    (embedding_size - cvm_offset),
                                    cvm_offset);
  }
}

template <typename T, typename Context>
void FusedSeqpoolCVMCUDAKernel(const Context &dev_ctx,
                               const std::vector<const DenseTensor *> &x,
                               const DenseTensor &cvm,
                               const std::string &pooltype,
                               float pad_value,
                               bool use_cvm,
                               int cvm_offset,
                               std::vector<DenseTensor *> out) {
  // from InferShape
  const size_t num_inputs = x.size();
  std::vector<phi::DDim> outs_dims;
  outs_dims.resize(num_inputs);
  int batch_size_tmp = -1;
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto dims = x[i]->dims();
    int rank = dims.size();
    int cur_batch_size = 0;

    const auto &x_lod = x[0]->lod();
    if (!x_lod.empty()) {
      cur_batch_size = static_cast<int>(x_lod[0].size() - 1);
    } else {
      cur_batch_size = static_cast<int>(x[0]->dims()[0]);
    }
    if (batch_size_tmp == -1) {
      batch_size_tmp = cur_batch_size;
    } else {
      PADDLE_ENFORCE_EQ(batch_size_tmp,
                        cur_batch_size,
                        common::errors::PreconditionNotMet(
                            "The batch size of all input should be same, "
                            "please check, last batch_size is %d, current "
                            "batch_size is %d",
                            batch_size_tmp,
                            cur_batch_size));
    }
    std::vector<int64_t> out_dim;
    if (use_cvm) {
      out_dim = {batch_size_tmp, dims[rank - 1]};
    } else {
      out_dim = {batch_size_tmp, dims[rank - 1] - cvm_offset};
    }
    outs_dims[i] = common::make_ddim(out_dim);
  }
  for (size_t i = 0; i < out.size(); ++i) {
    out[i]->Resize(outs_dims[i]);
  }

  auto &inputs = x;
  auto &outputs = out;
  const auto slot_size = inputs.size();
  std::vector<const float *> input_data(slot_size);
  std::vector<const size_t *> lods_data(slot_size);
  std::vector<T *> output_data(slot_size);

  std::vector<phi::DenseTensor> seqpool_outputs(slot_size);
  std::vector<T *> seqpool_output_data(slot_size);

  auto padding_value = pad_value;

  int embedding_size = inputs[0]->numel() / inputs[0]->dims()[0];
  int batch_size = -1;
  std::vector<phi::MixVector<size_t> *> mix_lods_v(slot_size);

  for (size_t i = 0; i < slot_size; ++i) {
    const auto *input = inputs[i];

    Vector<size_t> lods;
    if (input->lod().size() != 0) {
      auto lod = input->lod();
      lods = lod[0];
    } else {
      lods.push_back(0);
      for (int i = 0; i < input->dims()[0]; i++) {
        lods.push_back(i + 1);
      }
    }
    int cur_batch_size =
        input->lod().size() ? input->lod()[0].size() - 1 : input->dims()[0];
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
    input_data[i] = reinterpret_cast<const T *>(input->data<T>());

    auto *output = outputs[i];
    if (use_cvm) {
      output->Resize({batch_size, embedding_size});
    } else {
      output->Resize({batch_size, embedding_size - cvm_offset});
    }
    output_data[i] = reinterpret_cast<T *>(
        dev_ctx.template Alloc<T>(output, output->numel() * sizeof(T)));
    mix_lods_v[i] = new phi::MixVector<size_t>(&lods);
    lods_data[i] = mix_lods_v[i]->CUDAData(dev_ctx.GetPlace());
    seqpool_outputs[i].Resize({batch_size, embedding_size});
    seqpool_output_data[i] = reinterpret_cast<T *>(dev_ctx.template Alloc<T>(
        &seqpool_outputs[i], seqpool_outputs[i].numel() * sizeof(T)));
  }

  FusedSeqpoolCVM(dev_ctx,
                  input_data,
                  output_data,
                  seqpool_output_data,
                  lods_data,
                  batch_size,
                  slot_size,
                  embedding_size,
                  padding_value,
                  use_cvm,
                  cvm_offset);

  for (int i = 0; i < slot_size; i++) {
    delete mix_lods_v[i];
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_seqpool_cvm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedSeqpoolCVMCUDAKernel,
                   float) {}
