// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/new_executor/feed_fetch_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

namespace paddle::framework {

void SetColAttrForFeedFetchOps(std::shared_ptr<ProgramDesc> program_desc,
                               const int64_t micro_batch_num,
                               const int64_t micro_batch_id) {
  if (micro_batch_num < 2) return;

  const std::set<std::string>& valid_feed_fetch_op_types = {
      "fetch", "fetch_v2", "feed"};
  for (const auto& op_desc : program_desc->MutableBlock(0)->AllOps()) {
    if (valid_feed_fetch_op_types.find(op_desc->Type()) !=
        valid_feed_fetch_op_types.end()) {
      int col = op_desc->GetAttrIfExists<int>("col");
      PADDLE_ENFORCE_GE(
          col,
          0,
          common::errors::InvalidArgument(
              "Expected the column index (the attribute 'col' of "
              "operator 'Fetch') of current fetching variable to be "
              "no less than 0. But received column index = %d.",
              col));
      int new_col = static_cast<int>(col * micro_batch_num + micro_batch_id);
      op_desc->SetAttr("col", new_col);
      VLOG(6) << "Job (" << micro_batch_id << ") Set " << op_desc->Type()
              << "'s attr col=" << new_col;
    }
  }
}

void SplitFeedTensors(const std::vector<std::string>& feed_names,
                      const int64_t micro_batch_num,
                      Scope* scope,
                      std::vector<std::vector<phi::DenseTensor>>* out) {
  std::vector<phi::DenseTensor> feed_tensors;
  for (size_t i = 0; i < feed_names.size(); ++i) {
    auto feed_name = feed_names[i];
    auto feed_var = scope->GetVar(feed_name);
    PADDLE_ENFORCE_NOT_NULL(
        feed_var,
        common::errors::NotFound("Variable %s should not be nullptr.",
                                 feed_names[i]));
    feed_tensors.push_back(feed_var->Get<phi::DenseTensor>());
  }

  out->resize(micro_batch_num);
  if (micro_batch_num < 2) {
    (*out)[0] = std::move(feed_tensors);
    return;
  }

  for (size_t i = 0; i < feed_tensors.size(); ++i) {
    auto& feed_tensor = feed_tensors[i];
    int64_t numel_size = feed_tensor.dims()[0];
    PADDLE_ENFORCE_EQ(numel_size % micro_batch_num,
                      0,
                      common::errors::InvalidArgument(
                          "Split expects feed data (%s)'s dim[0] (%d) is "
                          "divisible by micro_batch_num (%d).",
                          feed_names[i],
                          numel_size,
                          micro_batch_num));
    int64_t split_size = numel_size / micro_batch_num;
    VLOG(4) << "Split feed data:" << feed_names[i] << ", dims:("
            << feed_tensor.dims() << "), micro_batch_num:" << micro_batch_num;
    for (int64_t j = 0; j < micro_batch_num; ++j) {
      (*out)[j].resize(i + 1);
      (*out)[j][i].ShareDataWith(
          feed_tensor.Slice(j * split_size, j * split_size + split_size));
    }
  }
}

void FetchTensors(const std::vector<std::string>& job_fetch_names,
                  const std::vector<std::string>& fetch_var_names,
                  const int64_t micro_batch_id,
                  Scope* scope,
                  FetchUnmergedList* fetch_list) {
  PADDLE_ENFORCE_GT(fetch_list->size(),
                    micro_batch_id,
                    common::errors::Unavailable(
                        "The fetch list size (%lld) should be greater "
                        "than micro_batch_id (%lld)",
                        fetch_list->size(),
                        micro_batch_id));

  fetch_list->at(micro_batch_id).resize(fetch_var_names.size());
  for (auto& var_name : job_fetch_names) {
    int col = find(fetch_var_names.begin(), fetch_var_names.end(), var_name) -
              fetch_var_names.begin();
    auto* var = scope->FindVar(var_name);
    if (var->IsType<phi::DenseTensor>()) {
      auto& src = var->Get<phi::DenseTensor>();
      auto* dst =
          &(PADDLE_GET(phi::DenseTensor, fetch_list->at(micro_batch_id)[col]));
      if (src.IsInitialized()) {
        TensorCopy(src, phi::CPUPlace(), dst);
        dst->set_lod(src.lod());
      } else {
        VLOG(6) << "Found " << var_name
                << " is not initialized and skip TensorCopy.";
      }
    } else if (var->IsType<phi::TensorArray>()) {
      auto& src = var->Get<phi::TensorArray>();
      fetch_list->at(micro_batch_id)[col] =
          phi::TensorArray();  // default DenseTensor, we replace it with
                               // TensorArray.
      auto* dst =
          &(PADDLE_GET(phi::TensorArray, fetch_list->at(micro_batch_id)[col]));
      dst->resize(src.size());
      for (size_t i = 0; i < src.size(); ++i) {
        TensorCopy(src[i], phi::CPUPlace(), &dst->at(i));
        dst->at(i).set_lod(src[i].lod());
      }
    }
  }
}

void MergeFetchTensors(const FetchUnmergedList& fetch_list,
                       const int64_t micro_batch_num,
                       FetchList* out) {
  if (fetch_list.size() == 0) return;

  PADDLE_ENFORCE_EQ(fetch_list.size(),
                    micro_batch_num,
                    common::errors::Unavailable(
                        "The fetch_list size (%lld) should be equal to "
                        "the micro_batch_num (%lld)",
                        fetch_list.size(),
                        micro_batch_num));

  if (micro_batch_num < 2) {
    *out = std::move(fetch_list[0]);
    return;
  }

  out->resize(fetch_list[0].size());
  for (size_t i = 0; i < fetch_list[0].size(); ++i) {
    std::vector<const phi::DenseTensor*> tensors_ptr;
    for (auto micro_batch_id = 0; micro_batch_id < micro_batch_num;
         ++micro_batch_id) {
      tensors_ptr.push_back(
          &PADDLE_GET_CONST(phi::DenseTensor, fetch_list[micro_batch_id][i]));
    }
    phi::DenseTensor merged_tensor;
    MergeTensors(tensors_ptr, phi::CPUPlace(), &merged_tensor);
    out->at(i) = std::move(merged_tensor);
  }
}

void MergeTensors(const std::vector<const phi::DenseTensor*>& tensors,
                  const phi::Place dst_place,
                  phi::DenseTensor* target) {
  PADDLE_ENFORCE_EQ(
      tensors.empty(),
      false,
      common::errors::InvalidArgument("The tensors to be merged are empty."));

  DDim new_dim = tensors[0]->dims();
  proto::VarType::Type new_type = proto::VarType::FP32;
  phi::DataLayout new_layout = tensors[0]->layout();
  for (auto* t : tensors) {
    if (t->numel() && t->IsInitialized()) {
      new_dim = t->dims();
      new_type = framework::TransToProtoVarType(t->dtype());
      new_layout = t->layout();
      break;
    }
  }

  auto rank = tensors[0]->dims().size();
  if (rank == 0) {
    std::vector<int> init_shape = {1};
    new_dim = new_dim.reshape(init_shape);
  }

  for (size_t i = 1; i < tensors.size(); ++i) {
    auto* t = tensors[i];
    if (t->numel() && t->IsInitialized()) {
      PADDLE_ENFORCE_EQ(
          new_type,
          framework::TransToProtoVarType(t->dtype()),
          common::errors::InvalidArgument(
              "phi::DenseTensor data type does not match, expected type is %s, "
              "actual "
              "type is %s.",
              DataTypeToString(new_type),
              DataTypeToString(framework::TransToProtoVarType(t->dtype()))));
      PADDLE_ENFORCE_EQ(
          new_layout,
          t->layout(),
          common::errors::InvalidArgument(
              "phi::DenseTensor layout does not match, expected layout is %s, "
              "actual layout is %s.",
              common::DataLayoutToString(new_layout),
              common::DataLayoutToString(t->layout())));
      if (rank > 0) {
        auto tensor_dims = t->dims();
        PADDLE_ENFORCE_EQ(tensor_dims.size(),
                          new_dim.size(),
                          common::errors::InvalidArgument(
                              "dimensions of DenseTensor does not match"));
        for (int j = 1; j < t->dims().size(); j++) {
          PADDLE_ENFORCE_EQ(
              tensor_dims[j],
              new_dim[j],
              common::errors::InvalidArgument(
                  "DenseTensor.ddim[%d] should equal to %d, but is %d",
                  j,
                  new_dim[j],
                  tensor_dims[j]));
        }
        new_dim[0] += t->dims()[0];
      } else if (rank == 0) {
        auto tensor_dims = t->dims();
        PADDLE_ENFORCE_EQ(tensor_dims.size(),
                          0,
                          common::errors::InvalidArgument(
                              "dimensions of DenseTensor does not match"));
        PADDLE_ENFORCE_EQ(new_dim.size(),
                          1,
                          common::errors::InvalidArgument(
                              "dimensions of DenseTensor does not match"));
        new_dim[0] += 1;
      }
    }
  }

  target->Resize(new_dim);
  target->set_layout(new_layout);
  target->mutable_data(dst_place, TransToPhiDataType(new_type));

  int begin = 0;
  for (auto* src : tensors) {
    int src_dim = 1;
    if (src->dims()[0] > 0) {
      src_dim = src->dims()[0];
    }
    int end = static_cast<int>(begin + src_dim);
    if (end == begin) {
      continue;
    }
    auto dst = target->Slice(begin, end);
    TensorCopy(*src, dst_place, &dst);
    begin = end;
  }
}

}  // namespace paddle::framework
