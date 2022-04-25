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

// normal
template <typename T>
__global__ void FusedSeqpoolKernelNormal(const size_t N, T **input_values,
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
__global__ void FusedCVMKernelWithCVM(const size_t N, T **output_values,
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
__global__ void FusedCVMKernelNoCVM(const size_t N, T **output_values,
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
void FusedSeqpoolCVM(const framework::ExecutionContext
                         &ctx,  // const paddle::platform::Place &place,
                     const std::vector<const T *> &input_data,
                     const std::vector<T *> &output_data,
                     const std::vector<T *> &seqpool_output_data,
                     std::vector<const size_t *> lods, const int batch_size,
                     const int slot_num, const int embedding_size,
                     const float padding_value, const bool use_cvm,
                     const int cvm_offset) {
  auto stream =
      ctx.template device_context<platform::CUDADeviceContext>().stream();
  auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  size_t total_ptr_len = input_data.size() + output_data.size() +
                         seqpool_output_data.size() + lods.size();
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
  T **gpu_seqpool_output_values =
      reinterpret_cast<T **>(&gpu_output_values[output_data.size()]);
  platform::GpuMemcpyAsync(
      gpu_seqpool_output_values, seqpool_output_data.data(),
      seqpool_output_data.size() * sizeof(T *), hipMemcpyHostToDevice, stream);
  size_t **lods_values = reinterpret_cast<size_t **>(
      &gpu_seqpool_output_values[seqpool_output_data.size()]);
  platform::GpuMemcpyAsync(lods_values, lods.data(),
                           lods.size() * sizeof(size_t *),
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
  T **gpu_seqpool_output_values =
      reinterpret_cast<T **>(&gpu_output_values[output_data.size()]);
  platform::GpuMemcpyAsync(
      gpu_seqpool_output_values, seqpool_output_data.data(),
      seqpool_output_data.size() * sizeof(T *), cudaMemcpyHostToDevice, stream);
  size_t **lods_values = reinterpret_cast<size_t **>(
      &gpu_seqpool_output_values[seqpool_output_data.size()]);
  platform::GpuMemcpyAsync(lods_values, lods.data(),
                           lods.size() * sizeof(size_t *),
                           cudaMemcpyHostToDevice, stream);
#endif

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  platform::GpuLaunchConfig config = GetGpuLaunchConfig1D(dev_ctx, N);
  // first sum pool
  FusedSeqpoolKernelNormal<<<config.block_per_grid.x, config.thread_per_block.x,
                             0, stream>>>(
      N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
      embedding_size, padding_value);
  // second log
  if (use_cvm) {
    FusedCVMKernelWithCVM<<<config.block_per_grid.x, config.thread_per_block.x,
                            0, stream>>>(N, gpu_output_values,
                                         gpu_seqpool_output_values, batch_size,
                                         embedding_size, cvm_offset);
  } else {
    // not need show click input
    N = static_cast<size_t>(batch_size * slot_num *
                            (embedding_size - cvm_offset));
    platform::GpuLaunchConfig config = GetGpuLaunchConfig1D(dev_ctx, N);
    FusedCVMKernelNoCVM<<<config.block_per_grid.x, config.thread_per_block.x, 0,
                          stream>>>(N, gpu_output_values,
                                    gpu_seqpool_output_values, batch_size,
                                    (embedding_size - cvm_offset), cvm_offset);
  }
}

// join grad
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelWithCVM(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
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
__global__ void FusedSeqpoolCVMGradKernelWithShow(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
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
__global__ void FusedSeqpoolCVMGradKernelNoCVM(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
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
void FusedSeqpoolCVMGrad(const framework::ExecutionContext &ctx,
                         const std::vector<const T *> &out_grads_data,
                         const std::vector<T *> &in_grads_data,
                         const std::vector<const T *> &cvm_data,
                         const std::vector<const size_t *> &lods,
                         const int batch_size, const int slot_num,
                         const int embedding_size, const bool use_cvm,
                         const int cvm_offset) {
  auto stream =
      ctx.template device_context<platform::CUDADeviceContext>().stream();
  auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  size_t total_ptr_len = out_grads_data.size() + in_grads_data.size() +
                         cvm_data.size() + lods.size();
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

  size_t **lods_values =
      reinterpret_cast<size_t **>(&gpu_cvm_values[cvm_data.size()]);
  platform::GpuMemcpyAsync(lods_values, lods.data(),
                           lods.size() * sizeof(size_t *),
                           hipMemcpyHostToDevice, stream);
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

  size_t **lods_values =
      reinterpret_cast<size_t **>(&gpu_cvm_values[cvm_data.size()]);
  platform::GpuMemcpyAsync(lods_values, lods.data(),
                           lods.size() * sizeof(size_t *),
                           cudaMemcpyHostToDevice, stream);
#endif

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  auto config = GetGpuLaunchConfig1D(dev_ctx, N);
  if (use_cvm) {
    // join grad
    FusedSeqpoolCVMGradKernelWithCVM<<<config.block_per_grid.x,
                                       config.thread_per_block.x, 0, stream>>>(
        N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
        lods_values, batch_size, embedding_size, cvm_offset);
  } else {
    // update grad
    FusedSeqpoolCVMGradKernelNoCVM<<<config.block_per_grid.x,
                                     config.thread_per_block.x, 0, stream>>>(
        N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
        lods_values, batch_size, embedding_size, cvm_offset);
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
    std::vector<const size_t *> lods_data(slot_size);
    std::vector<T *> output_data(slot_size);

    std::vector<LoDTensor> seqpool_outputs(slot_size);
    std::vector<T *> seqpool_output_data(slot_size);

    auto padding_value = ctx.Attr<float>("pad_value");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");

    int embedding_size = inputs[0]->numel() / inputs[0]->dims()[0];
    int batch_size = -1;
    std::vector<paddle::framework::MixVector<size_t> *> mix_lods_v(slot_size);

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
        PADDLE_ENFORCE_EQ(batch_size, cur_batch_size,
                          platform::errors::PreconditionNotMet(
                              "The batch size of all input should be same, "
                              "please cheack, last batchsize is %d, current "
                              "batchsize is %d",
                              batch_size, cur_batch_size));
      }
      input_data[i] = reinterpret_cast<const T *>(input->data<T>());

      auto *output = outputs[i];
      if (use_cvm) {
        output->Resize({batch_size, embedding_size});
      } else {
        output->Resize({batch_size, embedding_size - cvm_offset});
      }
      output_data[i] =
          reinterpret_cast<T *>(output->mutable_data<T>(ctx.GetPlace()));
      mix_lods_v[i] = new paddle::framework::MixVector<size_t>(&lods);
      lods_data[i] = mix_lods_v[i]->CUDAData(ctx.GetPlace());
      seqpool_output_data[i] =
          reinterpret_cast<T *>(seqpool_outputs[i].mutable_data<T>(
              {batch_size, embedding_size}, ctx.GetPlace()));
    }

    FusedSeqpoolCVM(ctx, input_data, output_data, seqpool_output_data,
                    lods_data, batch_size, slot_size, embedding_size,
                    padding_value, use_cvm, cvm_offset);

    for (int i = 0; i < slot_size; i++) {
      delete mix_lods_v[i];
    }
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
    std::vector<const size_t *> lods_data(slot_size);

    int embedding_size = in_grads[0]->numel() / in_grads[0]->dims()[0];
    int batch_size = -1;
    std::vector<paddle::framework::MixVector<size_t> *> mix_lods_v(slot_size);

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
        PADDLE_ENFORCE_EQ(batch_size, cur_batch_size,
                          platform::errors::PreconditionNotMet(
                              "The batch size of all input should be same, "
                              "please cheack, last batchsize is %d, current "
                              "batchsize is %d",
                              batch_size, cur_batch_size));
      }

      auto *out_grad = out_grads[i];
      out_grads_data[i] = reinterpret_cast<const T *>(out_grad->data<T>());

      in_grads_data[i] =
          reinterpret_cast<T *>(in_grad->mutable_data<T>(ctx.GetPlace()));
      mix_lods_v[i] = new paddle::framework::MixVector<size_t>(&lods);
      lods_data[i] = mix_lods_v[i]->CUDAData(ctx.GetPlace());
      cvm_data[i] = reinterpret_cast<const T *>(cvm->data<T>());
    }
    FusedSeqpoolCVMGrad(ctx, out_grads_data, in_grads_data, cvm_data, lods_data,
                        batch_size, slot_size, embedding_size, use_cvm,
                        cvm_offset);

    for (int i = 0; i < slot_size; i++) {
      delete mix_lods_v[i];
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm,
                        ops::FusedSeqpoolCVMCUDAKernel<float>);

REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_grad,
                        ops::FusedSeqpoolCVMGradCUDAKernel<float>);
