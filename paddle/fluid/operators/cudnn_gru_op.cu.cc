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

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/cudnn_rnn_cache.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <typename T>
static void StartForwardGRU(const framework::ExecutionContext &ctx,
                            CudnnRNNCache *cudnn_rnn_cache,
                            const std::vector<int> &seq_len, int max_seq_len,
                            bool is_test) {
  auto *x = ctx.Input<LoDTensor>("Input");
  auto run_seq_len = x->dims()[0];
  auto *out = ctx.Output<Tensor>("Out");
  T *out_data = out->mutable_data<T>(ctx.GetPlace());
  const Tensor *init_h = ctx.Input<Tensor>("InitH");
  auto w = ctx.Input<Tensor>("W");
  const T *w_data = w->data<T>();
  auto *last_h = ctx.Output<Tensor>("last_h");

  auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  auto handle = dev_ctx.cudnn_handle();

  math::SetConstant<paddle::platform::CUDADeviceContext, T> zero;
  const T *init_h_data = nullptr;
  if (init_h->numel() != 0) {
    init_h_data = init_h->data<T>();
  }
  auto init_h_dims = init_h->dims();

  T *last_h_data = nullptr;
  if (last_h == nullptr) {
    Tensor last_h_temp;
    last_h_temp.mutable_data<T>(init_h_dims, ctx.GetPlace());
    zero(dev_ctx, &last_h_temp, static_cast<T>(0.0));

    last_h_data = last_h_temp.data<T>();
  } else {
    last_h_data = last_h->mutable_data<T>(ctx.GetPlace());
  }

  const T *x_data = x->data<T>();
  T *x_data_padding = nullptr;

  if (!seq_len.empty()) {
    VLOG(3) << "This is a cudnn gru lod case with batch size > 1!";
    framework::DDim x_dim = x->dims();
    int x_padding_size = max_seq_len * x_dim[1];
    PADDLE_ENFORCE(
        cudaMalloc(reinterpret_cast<void **>(&x_data_padding), x_padding_size));
    // do cudnn for each lod, maybe need refine.
    for (auto &len : seq_len) {
      int mem_size = len * x_dim[1];
      PADDLE_ENFORCE(cudaMemcpy(x_data_padding, x_data, mem_size * sizeof(T),
                                cudaMemcpyDeviceToDevice));
      x_data += mem_size;
      run_seq_len = len;
      if (is_test) {
        // for inference
        CUDNN_ENFORCE(platform::dynload::cudnnRNNForwardInference(
            handle, cudnn_rnn_cache->rnn_desc_, run_seq_len,
            cudnn_rnn_cache->x_desc_, x_data, cudnn_rnn_cache->hx_desc_,
            init_h_data, cudnn_rnn_cache->cx_desc_, nullptr,
            cudnn_rnn_cache->w_desc_, w_data, cudnn_rnn_cache->y_desc_,
            out_data, cudnn_rnn_cache->hy_desc_, last_h_data,
            cudnn_rnn_cache->cy_desc_, nullptr,
            cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
            cudnn_rnn_cache->workspace_size_));
      } else {
        // for train
        CUDNN_ENFORCE(platform::dynload::cudnnRNNForwardTraining(
            handle, cudnn_rnn_cache->rnn_desc_, run_seq_len,
            cudnn_rnn_cache->x_desc_, x_data, cudnn_rnn_cache->hx_desc_,
            init_h_data, cudnn_rnn_cache->cx_desc_, nullptr,
            cudnn_rnn_cache->w_desc_, w_data, cudnn_rnn_cache->y_desc_,
            out_data, cudnn_rnn_cache->hy_desc_, last_h_data,
            cudnn_rnn_cache->cy_desc_, nullptr,
            cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
            cudnn_rnn_cache->workspace_size_,
            cudnn_rnn_cache->reserve_data_.data<uint8_t>(),
            cudnn_rnn_cache->reserve_size_));
      }
      // out has same dim as in
      out_data += len * out->dims()[1];
    }
  } else {
    if (is_test) {
      // for inference
      CUDNN_ENFORCE(platform::dynload::cudnnRNNForwardInference(
          handle, cudnn_rnn_cache->rnn_desc_, run_seq_len,
          cudnn_rnn_cache->x_desc_, x_data, cudnn_rnn_cache->hx_desc_,
          init_h_data, cudnn_rnn_cache->cx_desc_, nullptr,
          cudnn_rnn_cache->w_desc_, w_data, cudnn_rnn_cache->y_desc_, out_data,
          cudnn_rnn_cache->hy_desc_, last_h_data, cudnn_rnn_cache->cy_desc_,
          nullptr, cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
          cudnn_rnn_cache->workspace_size_));
    } else {
      // for train
      CUDNN_ENFORCE(platform::dynload::cudnnRNNForwardTraining(
          handle, cudnn_rnn_cache->rnn_desc_, run_seq_len,
          cudnn_rnn_cache->x_desc_, x_data, cudnn_rnn_cache->hx_desc_,
          init_h_data, cudnn_rnn_cache->cx_desc_, nullptr,
          cudnn_rnn_cache->w_desc_, w_data, cudnn_rnn_cache->y_desc_, out_data,
          cudnn_rnn_cache->hy_desc_, last_h_data, cudnn_rnn_cache->cy_desc_,
          nullptr, cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
          cudnn_rnn_cache->workspace_size_,
          cudnn_rnn_cache->reserve_data_.data<uint8_t>(),
          cudnn_rnn_cache->reserve_size_));
    }
  }
}

template <typename T>
static void InitForwardGRUParam(const framework::ExecutionContext &ctx,
                                CudnnRNNCache **cudnn_rnn_cache,
                                int *max_seq_len, std::vector<int> *seq_len) {
  auto *x = ctx.Input<LoDTensor>("Input");
  auto *w = ctx.Input<Tensor>("W");

  size_t max_len = ctx.Attr<int>("max_len");
  float dropout_prob = ctx.Attr<float>("dropout_prob");
  bool is_bidirec = ctx.Attr<bool>("is_bidirec");
  int input_size = ctx.Attr<int>("input_size");
  int hidden_size = ctx.Attr<int>("hidden_size");
  int num_layers = ctx.Attr<int>("num_layers");

  auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  auto handle = dev_ctx.cudnn_handle();
  auto *cache_var = ctx.InputVar("Cache");
  if (!cache_var) {
    // The RAW type cache variable wouldn't be created and broadcasted on
    // multi-devices before the first running.
    // use parent scope to make cache persistable
    auto *scope = const_cast<framework::Scope *>(ctx.scope().parent());
    auto cache_var_name = ctx.Inputs("Cache")[0];
    cache_var = scope->Var(cache_var_name);
  }
  if (cache_var->IsInitialized()) {
    *cudnn_rnn_cache = const_cast<framework::Variable *>(cache_var)
                           ->GetMutable<CudnnRNNCache>();
  } else {
    *cudnn_rnn_cache = const_cast<framework::Variable *>(cache_var)
                           ->GetMutable<CudnnRNNCache>();
    std::random_device rnd;
    int seed = ctx.Attr<int>("seed");
    if (seed == -1) {
      seed = rnd();
    }

    framework::DDim x_dim = x->dims();
    auto input_w_numel = w->numel();
    int batch_size = 1;
    // if input not lod tensor
    if (x->lod().empty()) {
      batch_size = x_dim[1];
    } else {
      framework::LoD x_lod = x->lod();
      batch_size = x_lod[0].size() - 1;
      max_len = x_dim[0];
      // if batch size > 1, save each seq len.
      if (batch_size > 1) {
        for (int i = 0; i < x_lod[0].size() - 1; ++i) {
          int len = x_lod[0][i + 1] - x_lod[0][i];
          seq_len->push_back(len);
          if (len > *max_seq_len) *max_seq_len = len;
        }
        max_len = *max_seq_len;
        batch_size = 1;
      }
    }
    (*cudnn_rnn_cache)
        ->init(handle, ctx.GetPlace(), max_len, batch_size, input_size,
               hidden_size, num_layers, dropout_prob, is_bidirec, seed,
               input_w_numel, true);
  }
}

template <typename T>
class CudnnGRUGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    CudnnRNNCache *cudnn_rnn_cache = nullptr;
    int max_seq_len = 0;
    std::vector<int> seq_len;
    bool is_test = ctx.Attr<bool>("is_test");
    InitForwardGRUParam<T>(ctx, &cudnn_rnn_cache, &max_seq_len, &seq_len);
    StartForwardGRU<T>(ctx, cudnn_rnn_cache, seq_len, max_seq_len, is_test);
  }
};

template <typename T>
class CudnnGRUGPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input = ctx.Input<Tensor>("Input");
    auto *weight = ctx.Input<Tensor>("W");
    auto *init_h = ctx.Input<Tensor>("InitH");
    auto *out = ctx.Input<Tensor>("Out");

    auto *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *in_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto *weight_grad = ctx.Output<Tensor>(framework::GradVarName("W"));
    auto *init_h_grad = ctx.Output<Tensor>(framework::GradVarName("InitH"));
    auto *last_h_grad = ctx.Input<Tensor>(framework::GradVarName("last_h"));

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto *cache_var = ctx.InputVar("Cache");
    PADDLE_ENFORCE(cache_var->IsInitialized());
    CudnnRNNCache *cudnn_rnn_cache =
        const_cast<framework::Variable *>(cache_var)
            ->GetMutable<CudnnRNNCache>();

    auto input_dims = input->dims();
    auto init_h_dims = init_h->dims();
    in_grad->mutable_data<T>(ctx.GetPlace());
    weight_grad->mutable_data<T>(ctx.GetPlace());
    math::SetConstant<paddle::platform::CUDADeviceContext, T> zero;
    zero(dev_ctx, in_grad, static_cast<T>(0.0));
    zero(dev_ctx, weight_grad, static_cast<T>(0.0));

    T *init_h_grad_data = NULL;
    if (init_h_grad == nullptr) {
      Tensor init_h_grad_temp;
      init_h_grad_temp.mutable_data<T>(init_h_dims, ctx.GetPlace());
      zero(dev_ctx, &init_h_grad_temp, static_cast<T>(0.0));

      init_h_grad_data = init_h_grad_temp.data<T>();
    } else {
      init_h_grad->mutable_data<T>(init_h_dims, ctx.GetPlace());
      zero(dev_ctx, init_h_grad, static_cast<T>(0.0));
      init_h_grad_data = init_h_grad->data<T>();
    }

    const T *last_h_grad_data = NULL;
    if (last_h_grad == nullptr) {
      Tensor last_h_grad_temp;
      last_h_grad_temp.mutable_data<T>(init_h_dims, ctx.GetPlace());
      zero(dev_ctx, &last_h_grad_temp, static_cast<T>(0.0));

      last_h_grad_data = (const T *)last_h_grad_temp.data<T>();
    } else {
      last_h_grad_data = last_h_grad->data<T>();
    }

    const T *out_grad_data = NULL;
    if (out_grad == nullptr) {
      Tensor out_grad_temp;
      out_grad_temp.mutable_data<T>(out->dims(), ctx.GetPlace());
      zero(dev_ctx, &out_grad_temp, static_cast<T>(0.0));

      out_grad_data = (const T *)out_grad_temp.data<T>();
    } else {
      out_grad_data = out_grad->data<T>();
    }

    auto out_data = out->data<T>();
    auto weight_data = weight->data<T>();
    auto init_h_data = init_h->data<T>();
    auto in_grad_data = in_grad->data<T>();

    auto work_data = cudnn_rnn_cache->workspace_data_.data<uint8_t>();
    auto reserve_data = cudnn_rnn_cache->reserve_data_.data<uint8_t>();

    auto run_seq_len = input_dims[0];
    PADDLE_ENFORCE_LE((size_t)run_seq_len, cudnn_rnn_cache->max_length_,
                      "cudnn running seq_len CAN not greater max_lengh");
    CUDNN_ENFORCE(platform::dynload::cudnnRNNBackwardData(
        handle, cudnn_rnn_cache->rnn_desc_, run_seq_len,
        cudnn_rnn_cache->y_desc_, out_data, cudnn_rnn_cache->dy_desc_,
        out_grad_data, cudnn_rnn_cache->dhy_desc_, last_h_grad_data,
        cudnn_rnn_cache->dcy_desc_, nullptr, cudnn_rnn_cache->w_desc_,
        weight_data, cudnn_rnn_cache->hx_desc_, init_h_data,
        cudnn_rnn_cache->cx_desc_, nullptr, cudnn_rnn_cache->dx_desc_,
        in_grad_data, cudnn_rnn_cache->dhx_desc_, init_h_grad_data,
        cudnn_rnn_cache->dcx_desc_, nullptr, work_data,
        cudnn_rnn_cache->workspace_size_, reserve_data,
        cudnn_rnn_cache->reserve_size_));

    CUDNN_ENFORCE(platform::dynload::cudnnRNNBackwardWeights(
        handle, cudnn_rnn_cache->rnn_desc_, run_seq_len,
        cudnn_rnn_cache->x_desc_, input->data<T>(), cudnn_rnn_cache->hx_desc_,
        init_h->data<T>(), cudnn_rnn_cache->y_desc_, out->data<T>(),
        cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
        cudnn_rnn_cache->workspace_size_, cudnn_rnn_cache->dw_desc_,
        weight_grad->data<T>(), cudnn_rnn_cache->reserve_data_.data<uint8_t>(),
        cudnn_rnn_cache->reserve_size_));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(cudnn_gru, ops::CudnnGRUGPUKernel<float>);
REGISTER_OP_CUDA_KERNEL(cudnn_gru_grad, ops::CudnnGRUGPUGradKernel<float>);
