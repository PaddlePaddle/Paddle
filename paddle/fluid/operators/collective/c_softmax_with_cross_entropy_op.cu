/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_softmax_with_cross_entropy_op.h"

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"

namespace paddle {
namespace operators {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T, typename IndexT>
__global__ void MaskLabelByIndex(T* predicted_logits,
                                 const T* logit,
                                 const IndexT* label,
                                 const int start_index,
                                 const int end_index,
                                 const int64_t N,
                                 const int64_t D,
                                 const int nranks) {
  CUDA_KERNEL_LOOP(i, N) {
    auto real_label = label[i];
    PADDLE_ENFORCE((real_label < D * nranks) && (real_label >= 0),
                   "The index is out of bounds, "
                   "please check whether the value of label and "
                   "input meet the class number. It should "
                   "be less than [%d], but received [%d]",
                   D * nranks,
                   real_label);

    if (real_label >= start_index && real_label < end_index) {
      predicted_logits[i] = logit[i * D + real_label - start_index];
    }
  }
}

template <typename T, typename IndexT>
__global__ void MaskLabelByIndexGrad(T* logits_grad,
                                     const T* loss_grad,
                                     const IndexT* labels,
                                     const int start_index,
                                     const int end_index,
                                     const int64_t N,
                                     const int64_t D) {
  CUDA_KERNEL_LOOP(i, N * D) {
    auto row = i / D;
    auto col = i % D;
    if ((col + start_index) == labels[row]) {
      logits_grad[i] = (logits_grad[i] - static_cast<T>(1.0)) * loss_grad[row];
    } else {
      logits_grad[i] *= loss_grad[row];
    }
  }
}

template <typename T>
class CSoftmaxWithCrossEntropyOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const int rid = ctx.Attr<int>("ring_id");
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      CSoftmaxWithCrossEntropyProcessGroupFunctor<phi::GPUContext, T> functor_;
      functor_(ctx);
    } else {
      CSoftmaxWithCrossEntropyFunctor<phi::GPUContext, T> functor_;
      functor_(ctx);
    }
  }
};

template <typename T>
struct CSoftmaxWithCrossEntropyFunctor<phi::GPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx) {
    const phi::DenseTensor* logits = ctx.Input<phi::DenseTensor>("Logits");
    const phi::DenseTensor* labels = ctx.Input<phi::DenseTensor>("Label");
    phi::DenseTensor* softmax = ctx.Output<phi::DenseTensor>("Softmax");
    phi::DenseTensor* loss = ctx.Output<phi::DenseTensor>("Loss");

    const int rid = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int rank = ctx.Attr<int>("rank");

    const auto& place = ctx.GetPlace();
    const auto& comm = platform::NCCLCommContext::Instance().Get(rid, place);
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    // use global calculate stream
    const auto stream = static_cast<phi::GPUContext*>(
                            platform::DeviceContextPool::Instance().Get(place))
                            ->stream();

    // allocate memory on device.
    softmax->mutable_data<T>(place);
    loss->mutable_data<T>(place);

    const auto& logits_dims = logits->dims();
    const auto& labels_dims = labels->dims();

    const int axis = logits_dims.size() - 1;
    const int N = phi::funcs::SizeToAxis(axis, logits_dims);
    const int D = phi::funcs::SizeFromAxis(axis, logits_dims);

    phi::DenseTensor logits_2d, softmax_2d, loss_2d;
    logits_2d.ShareDataWith(*logits).Resize({N, D});
    softmax_2d.ShareDataWith(*softmax).Resize({N, D});
    loss_2d.ShareDataWith(*loss).Resize({N, 1});

    auto eigen_logits = phi::funcs::EigenMatrix<T>::From(logits_2d);
    auto eigen_softmax = phi::funcs::EigenMatrix<T>::From(softmax_2d);

    // step 1, obtain logit_max
    phi::DenseTensor logits_max;
    logits_max = ctx.AllocateTmpTensor<T, phi::GPUContext>({N, 1}, dev_ctx);
    void* logits_max_buff = logits_max.mutable_data<T>(place);

    auto eigen_logits_max = phi::funcs::EigenMatrix<T>::From(logits_max);
    Eigen::DSizes<int, 1> along_axis(1);
    eigen_logits_max.device(*dev_ctx.eigen_device()) =
        eigen_logits.maximum(along_axis);
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
        logits_max_buff,
        logits_max_buff,
        logits_max.numel(),
        platform::ToNCCLDataType(
            framework::TransToProtoVarType(logits_max.dtype())),
        ncclMax,
        comm->comm(),
        stream));

    // step 2, obtain logit - logit_max
    Eigen::DSizes<int, 2> batch_by_one(N, 1);
    Eigen::DSizes<int, 2> one_by_class(1, D);

    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_logits -
         eigen_logits_max.reshape(batch_by_one).broadcast(one_by_class))
            .unaryExpr(phi::funcs::ValueClip<T>());

    // step 3, obtain predict target
    phi::DenseTensor predicted_logits;
    predicted_logits =
        ctx.AllocateTmpTensor<T, phi::GPUContext>({N, 1}, dev_ctx);
    predicted_logits.mutable_data<T>(place);

    auto t = framework::EigenVector<T>::Flatten(predicted_logits);
    t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

    const int start_index = rank * D;
    const int end_index = start_index + D;

    int blocks = NumBlocks(N);
    int threads = kNumCUDAThreads;
    const auto& label_type = framework::TransToProtoVarType(labels->dtype());

    if (label_type == framework::proto::VarType::INT32) {
      MaskLabelByIndex<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(predicted_logits.data<T>(),
                                                     softmax_2d.data<T>(),
                                                     labels->data<int32_t>(),
                                                     start_index,
                                                     end_index,
                                                     N,
                                                     D,
                                                     nranks);
    } else if (label_type == framework::proto::VarType::INT64) {
      MaskLabelByIndex<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(predicted_logits.data<T>(),
                                                     softmax_2d.data<T>(),
                                                     labels->data<int64_t>(),
                                                     start_index,
                                                     end_index,
                                                     N,
                                                     D,
                                                     nranks);
    }

    void* predict_logits_buff = predicted_logits.mutable_data<T>(place);
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
        predict_logits_buff,
        predict_logits_buff,
        predicted_logits.numel(),
        platform::ToNCCLDataType(
            framework::TransToProtoVarType(predicted_logits.dtype())),
        ncclSum,
        comm->comm(),
        stream));

    // step 4, obtain exp(logit)
    eigen_softmax.device(*dev_ctx.eigen_device()) = eigen_softmax.exp();

    // step 5, obtain sum_exp_logits
    phi::DenseTensor sum_exp_logits;
    sum_exp_logits = ctx.AllocateTmpTensor<T, phi::GPUContext>({N, 1}, dev_ctx);
    void* sum_exp_logits_buff = sum_exp_logits.mutable_data<T>(place);

    auto eigen_sum_exp_logits =
        phi::funcs::EigenMatrix<T>::From(sum_exp_logits);
    eigen_sum_exp_logits.device(*dev_ctx.eigen_device()) =
        eigen_softmax.sum(along_axis);

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
        sum_exp_logits_buff,
        sum_exp_logits_buff,
        sum_exp_logits.numel(),
        platform::ToNCCLDataType(
            framework::TransToProtoVarType(sum_exp_logits.dtype())),
        ncclSum,
        comm->comm(),
        stream));

    auto eigen_loss = phi::funcs::EigenMatrix<T>::From(loss_2d);
    auto eigen_predicted_logits =
        phi::funcs::EigenMatrix<T>::From(predicted_logits);

    eigen_loss.device(*dev_ctx.eigen_device()) =
        (eigen_sum_exp_logits.log().unaryExpr(phi::funcs::TolerableValue<T>()) -
         eigen_predicted_logits)
            .unaryExpr(phi::funcs::TolerableValue<T>());

    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_softmax *
         eigen_sum_exp_logits.inverse().broadcast(one_by_class));
  }
};

template <typename T>
struct CSoftmaxWithCrossEntropyProcessGroupFunctor<phi::GPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx) {
    const phi::DenseTensor* logits = ctx.Input<phi::DenseTensor>("Logits");
    const phi::DenseTensor* labels = ctx.Input<phi::DenseTensor>("Label");
    phi::DenseTensor* softmax = ctx.Output<phi::DenseTensor>("Softmax");
    phi::DenseTensor* loss = ctx.Output<phi::DenseTensor>("Loss");

    const int rid = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int rank = ctx.Attr<int>("rank");

    const auto& place = ctx.GetPlace();
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    distributed::ProcessGroup* pg = map->get(rid);
    distributed::AllreduceOptions opts;
    opts.reduce_op = distributed::ReduceOp::MAX;

    // allocate memory on device.
    softmax->mutable_data<T>(place);
    loss->mutable_data<T>(place);

    const auto& logits_dims = logits->dims();
    const auto& labels_dims = labels->dims();

    const int axis = logits_dims.size() - 1;
    const int N = phi::funcs::SizeToAxis(axis, logits_dims);
    const int D = phi::funcs::SizeFromAxis(axis, logits_dims);

    phi::DenseTensor logits_2d, softmax_2d, loss_2d;
    logits_2d.ShareDataWith(*logits).Resize({N, D});
    softmax_2d.ShareDataWith(*softmax).Resize({N, D});
    loss_2d.ShareDataWith(*loss).Resize({N, 1});

    auto eigen_logits = phi::funcs::EigenMatrix<T>::From(logits_2d);
    auto eigen_softmax = phi::funcs::EigenMatrix<T>::From(softmax_2d);

    // step 1, obtain logit_max
    phi::DenseTensor logits_max;
    logits_max = ctx.AllocateTmpTensor<T, phi::GPUContext>({N, 1}, dev_ctx);

    auto eigen_logits_max = phi::funcs::EigenMatrix<T>::From(logits_max);
    Eigen::DSizes<int, 1> along_axis(1);
    eigen_logits_max.device(*dev_ctx.eigen_device()) =
        eigen_logits.maximum(along_axis);

    std::vector<phi::DenseTensor> in_out;
    in_out.push_back(logits_max);
    pg->AllReduce(in_out, in_out, opts)->Synchronize();

    // step 2, obtain logit - logit_max
    Eigen::DSizes<int, 2> batch_by_one(N, 1);
    Eigen::DSizes<int, 2> one_by_class(1, D);

    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_logits -
         eigen_logits_max.reshape(batch_by_one).broadcast(one_by_class))
            .unaryExpr(phi::funcs::ValueClip<T>());

    // step 3, obtain predict target
    phi::DenseTensor predicted_logits;
    predicted_logits =
        ctx.AllocateTmpTensor<T, phi::GPUContext>({N, 1}, dev_ctx);
    predicted_logits.mutable_data<T>(place);

    auto t = framework::EigenVector<T>::Flatten(predicted_logits);
    t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

    const int start_index = rank * D;
    const int end_index = start_index + D;

    int blocks = NumBlocks(N);
    int threads = kNumCUDAThreads;
    const auto& label_type = framework::TransToProtoVarType(labels->dtype());

    if (label_type == framework::proto::VarType::INT32) {
      MaskLabelByIndex<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(predicted_logits.data<T>(),
                                                     softmax_2d.data<T>(),
                                                     labels->data<int32_t>(),
                                                     start_index,
                                                     end_index,
                                                     N,
                                                     D,
                                                     nranks);
    } else if (label_type == framework::proto::VarType::INT64) {
      MaskLabelByIndex<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(predicted_logits.data<T>(),
                                                     softmax_2d.data<T>(),
                                                     labels->data<int64_t>(),
                                                     start_index,
                                                     end_index,
                                                     N,
                                                     D,
                                                     nranks);
    }

    in_out.clear();
    in_out.push_back(predicted_logits);
    opts.reduce_op = distributed::ReduceOp::SUM;
    pg->AllReduce(in_out, in_out, opts)->Synchronize();

    // step 4, obtain exp(logit)
    eigen_softmax.device(*dev_ctx.eigen_device()) = eigen_softmax.exp();

    // step 5, obtain sum_exp_logits
    phi::DenseTensor sum_exp_logits;
    sum_exp_logits = ctx.AllocateTmpTensor<T, phi::GPUContext>({N, 1}, dev_ctx);
    void* sum_exp_logits_buff = sum_exp_logits.mutable_data<T>(place);

    auto eigen_sum_exp_logits =
        phi::funcs::EigenMatrix<T>::From(sum_exp_logits);
    eigen_sum_exp_logits.device(*dev_ctx.eigen_device()) =
        eigen_softmax.sum(along_axis);

    in_out.clear();
    in_out.push_back(sum_exp_logits);
    opts.reduce_op = distributed::ReduceOp::SUM;
    pg->AllReduce(in_out, in_out, opts)->Synchronize();

    auto eigen_loss = phi::funcs::EigenMatrix<T>::From(loss_2d);
    auto eigen_predicted_logits =
        phi::funcs::EigenMatrix<T>::From(predicted_logits);

    eigen_loss.device(*dev_ctx.eigen_device()) =
        (eigen_sum_exp_logits.log().unaryExpr(phi::funcs::TolerableValue<T>()) -
         eigen_predicted_logits)
            .unaryExpr(phi::funcs::TolerableValue<T>());

    eigen_softmax.device(*dev_ctx.eigen_device()) =
        (eigen_softmax *
         eigen_sum_exp_logits.inverse().broadcast(one_by_class));
  }
};

template <typename T>
class CSoftmaxWithCrossEntropyGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const phi::DenseTensor* labels = context.Input<phi::DenseTensor>("Label");
    const phi::DenseTensor* loss_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Loss"));
    phi::DenseTensor* logit_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("Logits"));
    const phi::DenseTensor* softmax =
        context.Input<phi::DenseTensor>("Softmax");
    const int rank = context.Attr<int>("rank");
    auto& dev_ctx = context.template device_context<phi::GPUContext>();

    if (logit_grad != softmax) {
      framework::TensorCopy(
          *softmax, context.GetPlace(), context.device_context(), logit_grad);
    }
    const auto sofrmax_dims = softmax->dims();
    const int axis = sofrmax_dims.size() - 1;
    const int N = phi::funcs::SizeToAxis(axis, sofrmax_dims);
    const int D = phi::funcs::SizeFromAxis(axis, sofrmax_dims);

    phi::DenseTensor logit_grad_2d;
    logit_grad_2d.ShareDataWith(*logit_grad).Resize({N, D});

    int blocks = NumBlocks(N * D);
    int threads = kNumCUDAThreads;
    const auto& label_type = framework::TransToProtoVarType(labels->dtype());
    const int start_index = rank * D;
    const int end_index = start_index + D;

    if (label_type == framework::proto::VarType::INT32) {
      MaskLabelByIndexGrad<T, int32_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(logit_grad_2d.data<T>(),
                                                     loss_grad->data<T>(),
                                                     labels->data<int32_t>(),
                                                     start_index,
                                                     end_index,
                                                     N,
                                                     D);
    } else if (label_type == framework::proto::VarType::INT64) {
      MaskLabelByIndexGrad<T, int64_t>
          <<<blocks, threads, 0, dev_ctx.stream()>>>(logit_grad_2d.data<T>(),
                                                     loss_grad->data<T>(),
                                                     labels->data<int64_t>(),
                                                     start_index,
                                                     end_index,
                                                     N,
                                                     D);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    c_softmax_with_cross_entropy,
    ops::CSoftmaxWithCrossEntropyOpCUDAKernel<float>,
    ops::CSoftmaxWithCrossEntropyOpCUDAKernel<double>,
    ops::CSoftmaxWithCrossEntropyOpCUDAKernel<plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    c_softmax_with_cross_entropy_grad,
    ops::CSoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::CSoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>,
    ops::CSoftmaxWithCrossEntropyGradCUDAKernel<double>);
