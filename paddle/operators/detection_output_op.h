/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/op_registry.h"
#include "paddle/framework/tensor.h"
#include "paddle/operators/math/detection_util.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/softmax.h"

namespace paddle {
namespace operators {
template <typename Place, typename T>
class Detection_output_Kernel : public framework::OpKernel<T> {
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

    int input_num = in_loc->dims()[0];
    int batch_size = in_loc->dims()[1];
    int channels = in_loc->dims()[2];
    int height = in_loc->dims()[3];
    int weight = in_loc->dims()[4];
    int loc_sum_size = in_loc->numel();
    int conf_sum_size = in_conf->numel();
    std::vector<int64_t> loc_shape_vec({1, loc_sum_size});
    std::vector<int64_t> conf_shape_vec(
        {conf_sum_size / num_classes, num_classes});
    framework::DDim loc_shape(framework::make_ddim(loc_shape_vec));
    framework::DDim conf_shape(framework::make_ddim(conf_shape_vec));
    framework::Tensor loc_tensor;
    framework::Tensor conf_tensor;
    loc_tensor.Resize(loc_shape);
    conf_tensor.Resize(conf_shape);
    loc_tensor.mutable_data<T>(loc_shape, context.GetPlace());
    conf_tensor.mutable_data<T>(conf_shape, context.GetPlace());
    framework::Tensor loc_cpu;
    framework::Tensor conf_cpu;
    framework::Tensor priorbox_cpu;
    const T* in_loc_data = in_loc->data<T>();
    const T* in_conf_data = in_conf->data<T>();
    T* loc_data;
    T* conf_data;
    const T* priorbox_data = in_priorbox->data<T>();

    if (platform::is_gpu_place(context.GetPlace())) {
      loc_cpu.mutable_data<T>(in_loc->dims(), platform::CPUPlace());
      framework::CopyFrom(*in_loc, platform::CPUPlace(),
                          context.device_context(), &loc_cpu);
      in_loc_data = loc_cpu.data<T>();
      conf_cpu.mutable_data<T>(in_conf->dims(), platform::CPUPlace());
      framework::CopyFrom(*in_conf, platform::CPUPlace(),
                          context.device_context(), &conf_cpu);
      in_conf_data = conf_cpu.data<T>();
      priorbox_cpu.mutable_data<T>(in_priorbox->dims(), platform::CPUPlace());
      framework::CopyFrom(*in_priorbox, platform::CPUPlace(),
                          context.device_context(), &priorbox_cpu);
      priorbox_data = priorbox_cpu.data<T>();
      loc_tensor.mutable_data<T>(loc_shape, platform::CPUPlace());
      conf_tensor.mutable_data<T>(conf_shape, platform::CPUPlace());
    }
    T* loc_tensor_data = loc_tensor.data<T>();
    T* conf_tensor_data = conf_tensor.data<T>();
    for (int i = 0; i < input_num; ++i) {
      math::appendWithPermute<T>(in_loc_data, input_num, batch_size, channels,
                                 height, weight, loc_tensor_data);
      math::appendWithPermute<T>(in_conf_data, input_num, batch_size, channels,
                                 height, weight, conf_tensor_data);
    }
    loc_data = loc_tensor.data<T>();
    if (platform::is_gpu_place(context.GetPlace())) {
      framework::Tensor conf_gpu;
      conf_gpu.Resize(conf_shape);
      conf_gpu.mutable_data<T>(conf_shape, context.GetPlace());
      framework::CopyFrom(conf_tensor, platform::GPUPlace(),
                          context.device_context(), &conf_gpu);
      // softmax
      math::SoftmaxFunctor<Place, T>()(context.device_context(), &conf_gpu,
                                       &conf_gpu);
      conf_tensor.mutable_data<T>(conf_gpu.dims(), platform::CPUPlace());
      framework::CopyFrom(conf_gpu, platform::CPUPlace(),
                          context.device_context(), &conf_tensor);
    } else {
      // softmax
      math::SoftmaxFunctor<Place, T>()(context.device_context(), &conf_tensor,
                                       &conf_tensor);
    }
    conf_data = conf_tensor.data<T>();
    // get decode bboxes
    size_t num_priors = in_priorbox->numel() / 8;
    std::vector<std::vector<operators::math::BBox<T>>> all_decoded_bboxes;
    for (size_t n = 0; n < batch_size; ++n) {
      std::vector<operators::math::BBox<T>> decoded_bboxes;
      for (size_t i = 0; i < num_priors; ++i) {
        size_t prior_offset = i * 8;
        size_t loc_pred_offset = n * num_priors * 4 + i * 4;
        std::vector<math::BBox<T>> prior_bbox_vec;
        math::getBBoxFromPriorData<T>(priorbox_data + prior_offset, 1,
                                      prior_bbox_vec);
        std::vector<std::vector<T>> prior_bbox_var;
        math::getBBoxVarFromPriorData<T>(priorbox_data + prior_offset, 1,
                                         prior_bbox_var);
        std::vector<T> loc_pred_data;
        for (size_t j = 0; j < 4; ++j)
          loc_pred_data.push_back(*(loc_data + loc_pred_offset + j));
        math::BBox<T> bbox = math::decodeBBoxWithVar<T>(
            prior_bbox_vec[0], prior_bbox_var[0], loc_pred_data);
        decoded_bboxes.push_back(bbox);
      }
      all_decoded_bboxes.push_back(decoded_bboxes);
    }
    std::vector<std::map<size_t, std::vector<size_t>>> all_indices;
    int num_kept = math::getDetectionIndices<T>(
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
    math::getDetectionOutput<T>(conf_data, num_kept, num_priors, num_classes,
                                batch_size, all_indices, all_decoded_bboxes,
                                out_data);
    if (platform::is_gpu_place(context.GetPlace())) {
      framework::CopyFrom(out_cpu, platform::GPUPlace(),
                          context.device_context(), out);
    }
  }
};
}  // namespace operators
}  // namespace paddle
