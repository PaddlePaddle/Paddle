/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   Indicesou may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/detection_util.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/strided_memcpy.h"
namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
inline void transpose_fun(const framework::ExecutionContext& context,
                          const framework::Tensor& src,
                          framework::Tensor* dst) {
  int input_nums = src.dims()[0];
  int offset = 0;
  for (int j = 0; j < input_nums; ++j) {
    framework::Tensor in_p_tensor = src.Slice(j, j + 1);
    std::vector<int64_t> shape_vec(
        {in_p_tensor.dims()[0], in_p_tensor.dims()[1], in_p_tensor.dims()[3],
         in_p_tensor.dims()[4], in_p_tensor.dims()[2]});
    framework::DDim shape(framework::make_ddim(shape_vec));
    framework::Tensor in_p_tensor_transpose;
    in_p_tensor_transpose.mutable_data<T>(shape, context.GetPlace());
    std::vector<int> shape_axis({0, 1, 3, 4, 2});
    math::Transpose<DeviceContext, T, 5> trans5;
    trans5(context.template device_context<DeviceContext>(), in_p_tensor,
           &in_p_tensor_transpose, shape_axis);
    auto dst_stride = framework::stride(dst->dims());
    auto src_stride = framework::stride(in_p_tensor_transpose.dims());
    StridedMemcpy<T>(context.device_context(), in_p_tensor_transpose.data<T>(),
                     src_stride, in_p_tensor_transpose.dims(), dst_stride,
                     dst->data<T>() + offset);
    offset += in_p_tensor_transpose.dims()[4] * src_stride[4];
  }
}
template <typename DeviceContext, typename T>
class DetectionOutputKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* in_loc = context.Input<framework::Tensor>("Loc");
    const framework::Tensor* in_conf = context.Input<framework::Tensor>("Conf");
    const framework::Tensor* in_priorbox =
        context.Input<framework::Tensor>("PriorBox");
    auto* out = context.Output<framework::Tensor>("Out");
    int num_classes = context.template Attr<int>("num_classes");
    int top_k = context.template Attr<int>("top_k");
    int nms_top_k = context.template Attr<int>("nms_top_k");
    int background_label_id = context.template Attr<int>("background_label_id");
    float nms_threshold = context.template Attr<float>("nms_threshold");
    float confidence_threshold =
        context.template Attr<float>("confidence_threshold");
    size_t batch_size = in_conf->dims()[1];
    int conf_sum_size = in_conf->numel();
    // for softmax
    std::vector<int64_t> conf_shape_softmax_vec(
        {conf_sum_size / num_classes, num_classes});
    framework::DDim conf_shape_softmax(
        framework::make_ddim(conf_shape_softmax_vec));
    // for knchw => nhwc
    std::vector<int64_t> loc_shape_vec({1, in_loc->dims()[1], in_loc->dims()[3],
                                        in_loc->dims()[4],
                                        in_loc->dims()[2] * in_loc->dims()[0]});
    std::vector<int64_t> conf_shape_vec(
        {1, in_conf->dims()[1], in_conf->dims()[3], in_conf->dims()[4],
         in_conf->dims()[2] * in_conf->dims()[0]});
    framework::DDim loc_shape(framework::make_ddim(loc_shape_vec));
    framework::DDim conf_shape(framework::make_ddim(conf_shape_vec));
    framework::Tensor loc_tensor;
    framework::Tensor conf_tensor;
    loc_tensor.mutable_data<T>(loc_shape, context.GetPlace());
    conf_tensor.mutable_data<T>(conf_shape, context.GetPlace());
    // for cpu
    framework::Tensor loc_cpu;
    framework::Tensor conf_cpu;
    framework::Tensor priorbox_cpu;
    const T* priorbox_data = in_priorbox->data<T>();
    transpose_fun<DeviceContext, T>(context, *in_loc, &loc_tensor);
    transpose_fun<DeviceContext, T>(context, *in_conf, &conf_tensor);
    conf_tensor.Resize(conf_shape_softmax);
    math::SoftmaxFunctor<DeviceContext, T>()(
        context.template device_context<DeviceContext>(), &conf_tensor,
        &conf_tensor);
    T* loc_data = loc_tensor.data<T>();
    T* conf_data = conf_tensor.data<T>();
    if (platform::is_gpu_place(context.GetPlace())) {
      loc_cpu.mutable_data<T>(loc_tensor.dims(), platform::CPUPlace());
      framework::TensorCopy(loc_tensor, platform::CPUPlace(),
                            context.device_context(), &loc_cpu);
      loc_data = loc_cpu.data<T>();
      conf_cpu.mutable_data<T>(conf_tensor.dims(), platform::CPUPlace());
      framework::TensorCopy(conf_tensor, platform::CPUPlace(),
                            context.device_context(), &conf_cpu);
      conf_data = conf_cpu.data<T>();
      priorbox_cpu.mutable_data<T>(in_priorbox->dims(), platform::CPUPlace());
      framework::TensorCopy(*in_priorbox, platform::CPUPlace(),
                            context.device_context(), &priorbox_cpu);
      priorbox_data = priorbox_cpu.data<T>();
    }
    // get decode bboxes
    size_t num_priors = in_priorbox->numel() / 8;
    std::vector<std::vector<operators::math::BBox<T>>> all_decoded_bboxes;
    for (size_t n = 0; n < batch_size; ++n) {
      std::vector<operators::math::BBox<T>> decoded_bboxes;
      for (size_t i = 0; i < num_priors; ++i) {
        size_t prior_offset = i * 8;
        size_t loc_pred_offset = n * num_priors * 4 + i * 4;
        std::vector<math::BBox<T>> prior_bbox_vec;
        math::GetBBoxFromPriorData<T>(priorbox_data + prior_offset, 1,
                                      prior_bbox_vec);
        std::vector<std::vector<T>> prior_bbox_var;
        math::GetBBoxVarFromPriorData<T>(priorbox_data + prior_offset, 1,
                                         prior_bbox_var);
        std::vector<T> loc_pred_data;
        for (size_t j = 0; j < 4; ++j)
          loc_pred_data.push_back(*(loc_data + loc_pred_offset + j));
        math::BBox<T> bbox = math::DecodeBBoxWithVar<T>(
            prior_bbox_vec[0], prior_bbox_var[0], loc_pred_data);
        decoded_bboxes.push_back(bbox);
      }
      all_decoded_bboxes.push_back(decoded_bboxes);
    }
    std::vector<std::map<size_t, std::vector<size_t>>> all_indices;
    int num_kept = math::GetDetectionIndices<T>(
        conf_data, num_priors, num_classes, background_label_id, batch_size,
        confidence_threshold, nms_top_k, nms_threshold, top_k,
        all_decoded_bboxes, &all_indices);

    if (num_kept <= 0) {
      std::vector<int64_t> out_shape_vec({0, 0});
      framework::DDim out_shape(framework::make_ddim(out_shape_vec));
      out->Resize(out_shape);
      return;
    }
    std::vector<int64_t> out_shape_vec({num_kept, 7});
    framework::DDim out_shape(framework::make_ddim(out_shape_vec));
    out->mutable_data<T>(out_shape, context.GetPlace());
    framework::Tensor out_cpu;
    T* out_data = out->data<T>();
    if (platform::is_gpu_place(context.GetPlace())) {
      out_cpu.mutable_data<T>(out->dims(), platform::CPUPlace());
      out_data = out_cpu.data<T>();
    }
    math::GetDetectionOutput<T>(conf_data, num_kept, num_priors, num_classes,
                                batch_size, all_indices, all_decoded_bboxes,
                                out_data);
    if (platform::is_gpu_place(context.GetPlace())) {
      framework::TensorCopy(out_cpu, platform::CUDAPlace(),
                            context.device_context(), out);
    }
  }
};
}  // namespace operators
}  // namespace paddle
