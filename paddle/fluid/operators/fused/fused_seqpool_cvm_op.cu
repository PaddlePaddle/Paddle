//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/operators/fused/fused_seqpool_cvm_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"

namespace paddle {
namespace operators {

template <typename T>
using Vector = framework::Vector<T>;

#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T>
__device__ void FusedCVMKernelNoCVMDevice(const size_t slot_num, T **input_values, T **output_values,
                                    const size_t *lods_values,
                                    const int batch_size,
                                    const float pad_value,
                                    const int embedding_size,
                                    const int cvm_offset,
                                    const int output_embedding_size,
                                    const int output_offset) {
  const size_t no_cvm_embedding_size = (embedding_size - cvm_offset);
  size_t N = no_cvm_embedding_size * slot_num * batch_size;
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / no_cvm_embedding_size;
    int offset = i % no_cvm_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    // no cvm
    auto start = lods_values[key + x]; // x * (batch_size + 1) + y
    auto end = lods_values[key + x + 1]; // x * (batch_size + 1) + y + 1

    T val = static_cast<T>(pad_value);
    auto input_values_ptr = input_values[x];
    for (auto k = start; k < end; ++k) {
      val += input_values_ptr[k * embedding_size + offset + cvm_offset];
    }
    output_values[x][y * output_embedding_size + offset + output_offset] = val;
  }
}

// join need show click input
template <typename T>
__global__ void FusedCVMKernelWithCVM(const size_t slot_num, T **input_values, T **output_values,
                                      const size_t *lods_values,
                                      const int batch_size,
                                      const float pad_value,
                                      const int embedding_size,
                                      const int cvm_offset) {
  FusedCVMKernelNoCVMDevice(slot_num,
    input_values, output_values, lods_values,
    batch_size, pad_value, embedding_size, cvm_offset, embedding_size, cvm_offset);
  size_t N = slot_num * batch_size;
  CUDA_KERNEL_LOOP(i, N) {
    int x = i / batch_size;  // slot id
    int y = i % batch_size;  // ins id
    auto start = lods_values[i + x]; // x * (batch_size + 1) + y
    auto end = lods_values[i + x + 1]; // x * (batch_size + 1) + y + 1
    T show_sum = 0;
    T click_sum = 0;
    auto input_values_ptr = input_values[x];
    for (auto k = start; k < end; ++k) {
      auto values_ptr = input_values_ptr + k * embedding_size;
      show_sum += values_ptr[0];
      click_sum += values_ptr[1];
    }
    show_sum = log(show_sum + 1);
    click_sum = show_sum - log(click_sum + 1);
    auto out_values_ptr = output_values[x] + y * embedding_size;
    out_values_ptr[0] = show_sum;
    out_values_ptr[1] = click_sum;
  }
}

// update not need show click input
template <typename T>
__global__ void FusedCVMKernelNoCVM(const size_t slot_num, T **input_values, T **output_values,
                                    const size_t *lods_values,
                                    const int batch_size,
                                    const float pad_value,
                                    const int embedding_size,
                                    const int cvm_offset) {
  FusedCVMKernelNoCVMDevice(slot_num,
    input_values, output_values, lods_values,
    batch_size, pad_value, embedding_size, cvm_offset, embedding_size - cvm_offset, 0);
}

template <typename T>
void FusedSeqpoolCVM(const framework::ExecutionContext
                         &ctx,  // const paddle::platform::Place &place,
                     const std::vector<const T *> &input_data,
                     const std::vector<T *> &output_data,
                     const size_t* lods, const int batch_size,
                     const int slot_num, const int embedding_size,
                     const float padding_value, const bool use_cvm,
                     const int cvm_offset) {
  auto stream =
      ctx.template device_context<platform::CUDADeviceContext>().stream();
  auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  size_t total_ptr_len = input_data.size() + output_data.size();
  auto temp_ptr =
      memory::AllocShared(ctx.GetPlace(), total_ptr_len * sizeof(void *));
  void *ptr = temp_ptr->ptr();

#ifdef PADDLE_WITH_HIP
  T **gpu_input_values = reinterpret_cast<T **>(temp_ptr->ptr());
  platform::GpuMemcpyAsync(gpu_input_values, input_data.data(),
                           input_data.size() * sizeof(T *),
                           hipMemcpyHostToDevice, stream);
  T **gpu_output_values =
      reinterpret_cast<T **>(&gpu_input_values[input_data.size()]);
  platform::GpuMemcpyAsync(gpu_output_values, output_data.data(),
                           output_data.size() * sizeof(T *),
                           hipMemcpyHostToDevice, stream);
#else
  T **gpu_input_values = reinterpret_cast<T **>(temp_ptr->ptr());
  platform::GpuMemcpyAsync(gpu_input_values, input_data.data(),
                           input_data.size() * sizeof(T *),
                           cudaMemcpyHostToDevice, stream);
  T **gpu_output_values =
      reinterpret_cast<T **>(&gpu_input_values[input_data.size()]);
  platform::GpuMemcpyAsync(gpu_output_values, output_data.data(),
                           output_data.size() * sizeof(T *),
                           cudaMemcpyHostToDevice, stream);
#endif

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  platform::GpuLaunchConfig config = GetGpuLaunchConfig1D(dev_ctx, N);
  // second log
  if (use_cvm) {
    FusedCVMKernelWithCVM<<<config.block_per_grid.x, config.thread_per_block.x,
                            0, stream>>>(slot_num, gpu_input_values, gpu_output_values, lods, batch_size, padding_value,
                                         embedding_size, cvm_offset);
  } else {
    // not need show click input
    N = static_cast<size_t>(batch_size * slot_num *
                            (embedding_size - cvm_offset));
    platform::GpuLaunchConfig config = GetGpuLaunchConfig1D(dev_ctx, N);
    FusedCVMKernelNoCVM<<<config.block_per_grid.x, config.thread_per_block.x, 0,
                          stream>>>(slot_num, gpu_input_values, gpu_output_values, lods, batch_size, padding_value,
                                    embedding_size, cvm_offset);
  }
}

// join grad
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelWithCVM(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    const size_t *lods_values, const int batch_size, const int embedding_size,
    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    T val = (offset < cvm_offset)
                 ? cvm_values[x][y * cvm_offset + offset]
                 : out_grads_values[x][y * embedding_size + offset];

    auto start = lods_values[key + x]; // x * (batch_size + 1) + y
    auto end = lods_values[key + x + 1]; // x * (batch_size + 1) + y + 1
    for (auto k = start; k < end; ++k) {
      in_grads_values[x][k * embedding_size + offset] = val;
    }
  }
}

// join only show not has click
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelWithShow(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    const size_t *lods_values, const int batch_size, const int embedding_size,
    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    T val =
        (offset < cvm_offset)
            ? cvm_values[x][y * cvm_offset + offset]
            : out_grads_values[x][y * (embedding_size - 1) + offset - 1];

    auto start = lods_values[key + x]; // x * (batch_size + 1) + y
    auto end = lods_values[key + x + 1]; // x * (batch_size + 1) + y + 1
    for (auto k = start; k < end; ++k) {
      in_grads_values[x][k * embedding_size + offset] = val;
    }
  }
}

// update grad
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelNoCVM(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    const size_t *lods_values, const int batch_size, const int embedding_size,
    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    T val = (offset < cvm_offset)
                 ? cvm_values[x][y * cvm_offset + offset]
                 : out_grads_values[x][y * (embedding_size - cvm_offset) +
                     offset - cvm_offset];

    auto start = lods_values[key + x]; // x * (batch_size + 1) + y
    auto end = lods_values[key + x + 1]; // x * (batch_size + 1) + y + 1
    for (auto k = start; k < end; ++k) {
      in_grads_values[x][k * embedding_size + offset] = val;
    }
  }
}

template <typename T>
void FusedSeqpoolCVMGrad(const framework::ExecutionContext &ctx,
                         const std::vector<const T *> &out_grads_data,
                         const std::vector<T *> &in_grads_data,
                         const std::vector<const T *> &cvm_data,
                         const size_t* lods,
                         const int batch_size, const int slot_num,
                         const int embedding_size, const bool use_cvm,
                         const int cvm_offset) {
  auto stream =
      ctx.template device_context<platform::CUDADeviceContext>().stream();
  auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  size_t total_ptr_len = out_grads_data.size() + in_grads_data.size() +
                         cvm_data.size();
  auto temp_ptr =
      memory::AllocShared(ctx.GetPlace(), total_ptr_len * sizeof(void *));
#ifdef PADDLE_WITH_HIP
  T **gpu_out_grads_values = reinterpret_cast<T **>(temp_ptr->ptr());
  platform::GpuMemcpyAsync(gpu_out_grads_values, out_grads_data.data(),
                           out_grads_data.size() * sizeof(T *),
                           hipMemcpyHostToDevice, stream);

  T **gpu_in_grads_values =
      reinterpret_cast<T **>(&gpu_out_grads_values[out_grads_data.size()]);
  platform::GpuMemcpyAsync(gpu_in_grads_values, in_grads_data.data(),
                           in_grads_data.size() * sizeof(T *),
                           hipMemcpyHostToDevice, stream);

  T **gpu_cvm_values =
      reinterpret_cast<T **>(&gpu_in_grads_values[in_grads_data.size()]);
  platform::GpuMemcpyAsync(gpu_cvm_values, cvm_data.data(),
                           cvm_data.size() * sizeof(T *), hipMemcpyHostToDevice,
                           stream);
#else
  T **gpu_out_grads_values = reinterpret_cast<T **>(temp_ptr->ptr());
  platform::GpuMemcpyAsync(gpu_out_grads_values, out_grads_data.data(),
                           out_grads_data.size() * sizeof(T *),
                           cudaMemcpyHostToDevice, stream);

  T **gpu_in_grads_values =
      reinterpret_cast<T **>(&gpu_out_grads_values[out_grads_data.size()]);
  platform::GpuMemcpyAsync(gpu_in_grads_values, in_grads_data.data(),
                           in_grads_data.size() * sizeof(T *),
                           cudaMemcpyHostToDevice, stream);

  T **gpu_cvm_values =
      reinterpret_cast<T **>(&gpu_in_grads_values[in_grads_data.size()]);
  platform::GpuMemcpyAsync(gpu_cvm_values, cvm_data.data(),
                           cvm_data.size() * sizeof(T *),
                           cudaMemcpyHostToDevice, stream);
#endif

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  auto config = GetGpuLaunchConfig1D(dev_ctx, N);
  if (use_cvm) {
    // join grad
    FusedSeqpoolCVMGradKernelWithCVM<<<config.block_per_grid.x,
                                       config.thread_per_block.x, 0, stream>>>(
        N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
        lods, batch_size, embedding_size, cvm_offset);
  } else {
    // update grad
    FusedSeqpoolCVMGradKernelNoCVM<<<config.block_per_grid.x,
                                     config.thread_per_block.x, 0, stream>>>(
        N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
        lods, batch_size, embedding_size, cvm_offset);
  }
}

template <typename T>
class FusedSeqpoolCVMCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<LoDTensor>("X");
    auto outputs = ctx.MultiOutput<framework::Tensor>("Out");

    const auto slot_size = inputs.size();
    std::vector<const float *> input_data(slot_size);
    std::vector<T *> output_data(slot_size);

    std::vector<LoDTensor> seqpool_outputs(slot_size);
    auto padding_value = ctx.Attr<float>("pad_value");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");

    int embedding_size = inputs[0]->numel() / inputs[0]->dims()[0];
    int batch_size = inputs[0]->lod().size() ? inputs[0]->lod()[0].size() - 1 : inputs[0]->dims()[0];
    std::vector<size_t> mix_lods;
    mix_lods.reserve(slot_size * (batch_size + 1));
    #pragma omp parallel for
    for (size_t i = 0; i < slot_size; ++i) {
      const auto *input = inputs[i];
      if (input->lod().size() != 0) {
        auto lod = input->lod();
        PADDLE_ENFORCE_EQ(lod.size(), 1,
                          platform::errors::PreconditionNotMet(
                              "The lod size of all input should be 1, "
                              "please cheack"));
        PADDLE_ENFORCE_EQ(lod[0].size(), batch_size + 1,
                          platform::errors::PreconditionNotMet(
                              "The lod[0] size of all input should be batch_size + 1, "
                              "please cheack"));
        mix_lods.insert(mix_lods.end(), lod[0].begin(), lod[0].end());
      } else {
        mix_lods.push_back(0);
        for (int i = 0; i < input->dims()[0]; i++) {
          mix_lods.push_back(i + 1);
        }
      }
      int cur_batch_size =
          input->lod().size() ? input->lod()[0].size() - 1 : input->dims()[0];
      PADDLE_ENFORCE_EQ(batch_size, cur_batch_size,
                        platform::errors::PreconditionNotMet(
                            "The batch size of all input should be same, "
                            "please cheack, last batchsize is %d, current "
                            "batchsize is %d",
                            batch_size, cur_batch_size));
    }
    PADDLE_ENFORCE_EQ(mix_lods.size(), slot_size * (batch_size + 1),
                      platform::errors::PreconditionNotMet(
                          "please cheack"));
    paddle::framework::MixVector<size_t> mix_lods_v(&mix_lods);
    auto mix_lods_data = mix_lods_v.CUDAData(ctx.GetPlace());
    #pragma omp parallel for
    for (size_t i = 0; i < slot_size; ++i) {
      const auto *input = inputs[i];
      input_data[i] = reinterpret_cast<const T *>(input->data<T>());

      auto *output = outputs[i];
      if (use_cvm) {
        output->Resize({batch_size, embedding_size});
      } else {
        output->Resize({batch_size, embedding_size - cvm_offset});
      }
      output_data[i] =
          reinterpret_cast<T *>(output->mutable_data<T>(ctx.GetPlace()));
    }

    FusedSeqpoolCVM(ctx, input_data, output_data,
                    mix_lods_data, batch_size, slot_size, embedding_size,
                    padding_value, use_cvm, cvm_offset);
  }
};

template <typename T>
class FusedSeqpoolCVMGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_grads = ctx.MultiInput<LoDTensor>(framework::GradVarName("Out"));
    auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));
    auto *cvm = ctx.Input<LoDTensor>("CVM");

    std::string pooltype = ctx.Attr<std::string>("pooltype");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");

    const auto slot_size = in_grads.size();
    std::vector<const T *> out_grads_data(slot_size);
    std::vector<T *> in_grads_data(slot_size);
    std::vector<const T *> cvm_data(slot_size);

    int embedding_size = in_grads[0]->numel() / in_grads[0]->dims()[0];
    int batch_size = in_grads[0]->lod().size() ? in_grads[0]->lod()[0].size() - 1 : in_grads[0]->dims()[0];
    std::vector<size_t> mix_lods;
    mix_lods.reserve(slot_size * (batch_size + 1));
    #pragma omp parallel for
    for (size_t i = 0; i < slot_size; ++i) {
      auto *in_grad = in_grads[i];
      if (in_grad->lod().size() != 0) {
        auto lod = in_grad->lod();
        PADDLE_ENFORCE_EQ(lod.size(), 1,
                          platform::errors::PreconditionNotMet(
                              "The lod size of all in_grad should be 1, "
                              "please cheack"));
        PADDLE_ENFORCE_EQ(lod[0].size(), batch_size + 1,
                          platform::errors::PreconditionNotMet(
                              "The lod[0] size of all in_grad should be batch_size + 1, "
                              "please cheack"));
        mix_lods.insert(mix_lods.end(), lod[0].begin(), lod[0].end());
      } else {
        mix_lods.push_back(0);
        for (int i = 0; i < in_grad->dims()[0]; i++) {
          mix_lods.push_back(i + 1);
        }
      }
      int cur_batch_size = in_grad->lod().size() ? in_grad->lod()[0].size() - 1 : in_grad->dims()[0];
      PADDLE_ENFORCE_EQ(batch_size, cur_batch_size,
                        platform::errors::PreconditionNotMet(
                            "The batch size of all in_grad should be same, "
                            "please cheack, last batchsize is %d, current "
                            "batchsize is %d",
                            batch_size, cur_batch_size));
    }
    PADDLE_ENFORCE_EQ(mix_lods.size(), slot_size * (batch_size + 1),
                      platform::errors::PreconditionNotMet(
                          "please cheack"));
    paddle::framework::MixVector<size_t> mix_lods_v(&mix_lods);
    auto mix_lods_data = mix_lods_v.CUDAData(ctx.GetPlace());

    #pragma omp parallel for
    for (size_t i = 0; i < slot_size; ++i) {
      auto *in_grad = in_grads[i];

      auto *out_grad = out_grads[i];
      out_grads_data[i] = reinterpret_cast<const T *>(out_grad->data<T>());

      in_grads_data[i] =
          reinterpret_cast<T *>(in_grad->mutable_data<T>(ctx.GetPlace()));
      cvm_data[i] = reinterpret_cast<const T *>(cvm->data<T>());
    }
    FusedSeqpoolCVMGrad(ctx, out_grads_data, in_grads_data, cvm_data, mix_lods_data,
                        batch_size, slot_size, embedding_size, use_cvm,
                        cvm_offset);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm,
                        ops::FusedSeqpoolCVMCUDAKernel<float>);

REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_grad,
                        ops::FusedSeqpoolCVMGradCUDAKernel<float>);
