/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class ROIAlignOpMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<phi::DenseTensor>("X");
    auto* rois = ctx.Input<phi::DenseTensor>("ROIs");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    out->set_layout(phi::DataLayout::kNHWC);

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<int>("sampling_ratio");
    auto aligned = ctx.Attr<bool>("aligned");
    const auto& in_dims = in->dims();
    int batch_size = in_dims[0];
    int rois_num = rois->dims()[0];

    if (rois_num == 0) return;
    auto cplace = platform::CPUPlace();
    std::vector<int> roi_batch_id_list(rois_num);
    int rois_batch_size = 0;
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<phi::DenseTensor>("RoisNum");
      rois_batch_size = rois_num_t->numel();
      PADDLE_ENFORCE_EQ(
          rois_batch_size,
          batch_size,
          platform::errors::InvalidArgument(
              "The batch size of rois and the batch size of images "
              " must be the same. But received the batch size of rois is %d, "
              "and the batch size of images is %d",
              rois_batch_size,
              batch_size));
      std::vector<int> rois_num_list(rois_batch_size);
      memory::Copy(cplace,
                   rois_num_list.data(),
                   ctx.GetPlace(),
                   rois_num_t->data<int>(),
                   sizeof(int) * rois_batch_size,
                   nullptr /*stream*/);
      int last_idx = 0;
      for (int i = 0; i < rois_batch_size; i++) {
        int end_idx = last_idx + rois_num_list[i];
        for (int j = last_idx; j < end_idx; j++) {
          roi_batch_id_list[j] = i;
        }
        last_idx = end_idx;
      }
    } else {
      auto lod = rois->lod();
      PADDLE_ENFORCE_EQ(lod.empty(),
                        false,
                        platform::errors::InvalidArgument(
                            "Input(ROIs) phi::DenseTensor of ROIAlignOp "
                            "does not contain LoD information."));
      auto rois_lod = lod.back();
      rois_batch_size = rois_lod.size() - 1;
      PADDLE_ENFORCE_EQ(rois_batch_size,
                        batch_size,
                        platform::errors::InvalidArgument(
                            "The rois_batch_size and imgs "
                            "batch_size must be the same. But received "
                            "rois_batch_size = %d, "
                            "batch_size = %d",
                            rois_batch_size,
                            batch_size));
      int rois_num_with_lod = rois_lod[rois_batch_size];
      PADDLE_ENFORCE_EQ(
          rois_num,
          rois_num_with_lod,
          platform::errors::InvalidArgument(
              "The actual number of rois and the number of rois "
              "provided from Input(RoIsLoD) in RoIAlign must be the same."
              " But received actual number of rois is %d, and the number "
              "of rois from RoIsLoD is %d",
              rois_num,
              rois_num_with_lod));
      for (int i = 0; i < rois_batch_size; i++) {
        int start_idx = rois_lod[i];
        int end_idx = rois_lod[i + 1];
        for (int j = start_idx; j < end_idx; j++) {
          roi_batch_id_list[j] = i;
        }
      }
    }

    // only support float32 for now
    phi::DenseTensor rois_cpu(framework::TransToPhiDataType(VT::FP32));
    rois_cpu.Resize({rois_num, 4});
    rois_cpu.mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();
    framework::TensorCopy(*rois, cplace, dev_ctx, &rois_cpu);
    dev_ctx.Wait();
    T* rois_cpu_ptr = rois_cpu.mutable_data<T>(platform::CPUPlace());

    // boxes; [batch_idx, x1, y1, x2, y2]
    phi::DenseTensor boxes_cpu(framework::TransToPhiDataType(VT::FP32));
    phi::DenseTensor boxes_mlu(framework::TransToPhiDataType(VT::FP32));
    boxes_cpu.Resize({rois_num, 5});
    boxes_mlu.Resize({rois_num, 5});
    T* boxes_cpu_ptr = boxes_cpu.mutable_data<T>(platform::CPUPlace());
    boxes_mlu.mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < rois_num; ++i) {
      boxes_cpu_ptr[i * 5 + 0] = static_cast<T>(roi_batch_id_list[i]);
      boxes_cpu_ptr[i * 5 + 1] = rois_cpu_ptr[i * 4 + 0];
      boxes_cpu_ptr[i * 5 + 2] = rois_cpu_ptr[i * 4 + 1];
      boxes_cpu_ptr[i * 5 + 3] = rois_cpu_ptr[i * 4 + 2];
      boxes_cpu_ptr[i * 5 + 4] = rois_cpu_ptr[i * 4 + 3];
    }

    // copy boxes_cpu to boxes_mlu
    framework::TensorCopy(boxes_cpu, ctx.GetPlace(), dev_ctx, &boxes_mlu);
    dev_ctx.Wait();

    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
    const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
    phi::DenseTensor input_nhwc(in->type());
    phi::DenseTensor output_nhwc(out->type());
    TransposeFromMLUTensor<T>(
        ctx, perm_to_nhwc, in, &input_nhwc, true /*need_reshape_or_alloc*/);
    auto output_dims = out->dims();
    output_nhwc.mutable_data<T>(
        {output_dims[0], output_dims[2], output_dims[3], output_dims[1]},
        ctx.GetPlace());

    MLUCnnlTensorDesc input_desc(
        input_nhwc, CNNL_LAYOUT_NHWC, ToCnnlDataType(input_nhwc.dtype()));
    MLUCnnlTensorDesc boxes_desc(boxes_mlu);
    MLUCnnlTensorDesc out_desc(
        output_nhwc, CNNL_LAYOUT_NHWC, ToCnnlDataType(output_nhwc.dtype()));
    MLUCnnl::RoiAlign(ctx,
                      pooled_height,
                      pooled_width,
                      sampling_ratio,
                      spatial_scale,
                      aligned,
                      input_desc.get(),
                      GetBasePtr(&input_nhwc),
                      boxes_desc.get(),
                      GetBasePtr(&boxes_mlu),
                      out_desc.get(),
                      GetBasePtr(&output_nhwc));
    TransposeFromMLUTensor<T>(
        ctx, perm_to_nchw, &output_nhwc, out, false /*need_reshape_or_alloc*/);
  };
};

template <typename T>
class ROIAlignGradOpMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* rois = ctx.Input<phi::DenseTensor>("ROIs");
    auto* out_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    auto spatial_scale = ctx.Attr<T>("spatial_scale");
    auto sampling_ratio = ctx.Attr<int>("sampling_ratio");
    auto aligned = ctx.Attr<bool>("aligned");
    int rois_num = rois->dims()[0];

    if (!in_grad) {
      return;
    }
    in_grad->mutable_data<T>(ctx.GetPlace());

    std::vector<int> roi_batch_id_list(rois_num);
    auto cplace = platform::CPUPlace();
    int rois_batch_size = 0;
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<phi::DenseTensor>("RoisNum");
      rois_batch_size = rois_num_t->numel();
      std::vector<int> rois_num_list(rois_batch_size);
      memory::Copy(cplace,
                   rois_num_list.data(),
                   ctx.GetPlace(),
                   rois_num_t->data<int>(),
                   sizeof(int) * rois_batch_size,
                   nullptr /*stream*/);
      int last_idx = 0;
      for (int i = 0; i < rois_batch_size; i++) {
        int end_idx = last_idx + rois_num_list[i];
        for (int j = last_idx; j < end_idx; j++) {
          roi_batch_id_list[j] = i;
        }
        last_idx = end_idx;
      }
    } else {
      auto rois_lod = rois->lod().back();
      rois_batch_size = rois_lod.size() - 1;
      for (int i = 0; i < rois_batch_size; i++) {
        int start_idx = rois_lod[i];
        int end_idx = rois_lod[i + 1];
        for (int j = start_idx; j < end_idx; j++) {
          roi_batch_id_list[j] = i;
        }
      }
    }

    phi::DenseTensor rois_cpu(framework::TransToPhiDataType(VT::FP32));
    rois_cpu.Resize({rois_num, 4});
    rois_cpu.mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();
    framework::TensorCopy(*rois, cplace, dev_ctx, &rois_cpu);
    dev_ctx.Wait();
    T* rois_cpu_ptr = rois_cpu.mutable_data<T>(platform::CPUPlace());

    // boxes; [batch_idx, x1, y1, x2, y2]
    phi::DenseTensor boxes_cpu(framework::TransToPhiDataType(VT::FP32));
    phi::DenseTensor boxes_mlu(framework::TransToPhiDataType(VT::FP32));
    boxes_cpu.Resize({rois_num, 5});
    boxes_mlu.Resize({rois_num, 5});
    T* boxes_cpu_ptr = boxes_cpu.mutable_data<T>(platform::CPUPlace());
    boxes_mlu.mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < rois_num; ++i) {
      boxes_cpu_ptr[i * 5 + 0] = static_cast<T>(roi_batch_id_list[i]);
      boxes_cpu_ptr[i * 5 + 1] = rois_cpu_ptr[i * 4 + 0];
      boxes_cpu_ptr[i * 5 + 2] = rois_cpu_ptr[i * 4 + 1];
      boxes_cpu_ptr[i * 5 + 3] = rois_cpu_ptr[i * 4 + 2];
      boxes_cpu_ptr[i * 5 + 4] = rois_cpu_ptr[i * 4 + 3];
    }

    // copy boxes_cpu to boxes_mlu
    framework::TensorCopy(boxes_cpu, ctx.GetPlace(), dev_ctx, &boxes_mlu);
    dev_ctx.Wait();

    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
    const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
    phi::DenseTensor grads_nhwc(out_grad->type());
    phi::DenseTensor grads_image_nhwc(in_grad->type());
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nhwc,
                              out_grad,
                              &grads_nhwc,
                              true /*need_reshape_or_alloc*/);
    auto grads_image_dims = in_grad->dims();
    grads_image_nhwc.mutable_data<T>({grads_image_dims[0],
                                      grads_image_dims[2],
                                      grads_image_dims[3],
                                      grads_image_dims[1]},
                                     ctx.GetPlace());

    MLUCnnlTensorDesc grads_desc(
        grads_nhwc, CNNL_LAYOUT_NHWC, ToCnnlDataType(grads_nhwc.dtype()));
    MLUCnnlTensorDesc boxes_desc(boxes_mlu);
    MLUCnnlTensorDesc grads_image_desc(
        grads_image_nhwc,
        CNNL_LAYOUT_NHWC,
        ToCnnlDataType(grads_image_nhwc.dtype()));
    MLUCnnl::RoiAlignBackward(ctx,
                              sampling_ratio,
                              spatial_scale,
                              aligned,
                              grads_desc.get(),
                              GetBasePtr(&grads_nhwc),
                              boxes_desc.get(),
                              GetBasePtr(&boxes_mlu),
                              grads_image_desc.get(),
                              GetBasePtr(&grads_image_nhwc));
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nchw,
                              &grads_image_nhwc,
                              in_grad,
                              false /*need_reshape_or_alloc*/);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(roi_align, ops::ROIAlignOpMLUKernel<float>);

REGISTER_OP_MLU_KERNEL(roi_align_grad, ops::ROIAlignGradOpMLUKernel<float>);
