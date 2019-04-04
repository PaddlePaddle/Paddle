// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/anakin/convert/roi_align.h"
#include <algorithm>
#include <map>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void RoiAlignOpConverter<TargetT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Input("ROIs").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();
  auto input_x_name = op_desc.Input("X").front();
  auto input_rois_name = op_desc.Input("ROIs").front();
  auto output_name = op_desc.Output("Out").front();

  auto spatial_scale = boost::get<float>(op_desc.GetAttr("spatial_scale"));
  auto pooled_height = boost::get<int>(op_desc.GetAttr("pooled_height"));
  auto pooled_width = boost::get<int>(op_desc.GetAttr("pooled_width"));
  auto sampling_ratio = boost::get<int>(op_desc.GetAttr("sampling_ratio"));

  this->engine_->AddOp(op_name, "RoiAlign", {input_x_name, input_rois_name},
                       {output_name});
  this->engine_->AddOpAttr(op_name, "spatial_scale", spatial_scale);
  this->engine_->AddOpAttr(op_name, "pooled_height", pooled_height);
  this->engine_->AddOpAttr(op_name, "pooled_width", pooled_width);
  this->engine_->AddOpAttr(op_name, "sampling_ratio", sampling_ratio);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

#ifdef PADDLE_WITH_CUDA
REGISTER_CUDA_ANAKIN_OP_CONVERTER(roi_align,
                                  RoiAlignOpConverter<::anakin::saber::NV>);
#endif
REGISTER_CPU_ANAKIN_OP_CONVERTER(roi_align,
                                 RoiAlignOpConverter<::anakin::saber::X86>);
