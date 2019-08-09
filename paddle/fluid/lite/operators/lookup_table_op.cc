// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/operators/lookup_table_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool LookupTableOp::CheckShape() const {
  CHECK_OR_FALSE(param_.w);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.ids);
  auto table_dims = param_.w->dims();
  auto ids_dims = param_.ids->dims();
  int ids_rank = ids_dims.size();
  CHECK_EQ_OR_FALSE(table_dims.size(), 2UL);
  CHECK_EQ_OR_FALSE(ids_dims[ids_rank - 1], 1UL);

  return true;
}

bool LookupTableOp::InferShape() const {
  auto table_dims = param_.w->dims();
  auto ids_dims = param_.ids->dims();
  int ids_rank = ids_dims.size();
  auto output_dims = framework::vectorize(
      framework::slice_ddim(ids_dims.data(), 0, ids_rank - 1));
  output_dims.push_back(table_dims[1]);
  param_.output->Resize(lite::DDim(output_dims));
  param_.output->raw_tensor().set_lod(param_.ids->raw_tensor().lod());
  return true;
}

bool LookupTableOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.w =
      scope->FindVar(opdesc.Input("W").front())->GetMutable<lite::Tensor>();
  param_.ids =
      scope->FindVar(opdesc.Input("Ids").front())->GetMutable<lite::Tensor>();
  param_.output =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();

  if (opdesc.HasAttr("is_sparse"))
    param_.is_sparse = opdesc.GetAttr<bool>("is_sparse");
  if (opdesc.HasAttr("is_distributed"))
    param_.is_distributed = opdesc.GetAttr<bool>("is_distributed");
  param_.padding_idx = opdesc.GetAttr<int64_t>("padding_idx");
  if (opdesc.HasAttr("remote_prefetch"))
    param_.remote_prefetch = opdesc.GetAttr<bool>("remote_prefetch");
  if (opdesc.HasAttr("trainer_id"))
    param_.trainer_id = opdesc.GetAttr<int>("trainer_id");
  if (opdesc.HasAttr("height_sections"))
    param_.height_sections =
        opdesc.GetAttr<std::vector<int64_t>>("height_sections");
  if (opdesc.HasAttr("epmap"))
    param_.epmap = opdesc.GetAttr<std::vector<std::string>>("epmap");
  if (opdesc.HasAttr("table_names"))
    param_.table_names =
        opdesc.GetAttr<std::vector<std::string>>("table_names");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(lookup_table, paddle::lite::operators::LookupTableOp);
