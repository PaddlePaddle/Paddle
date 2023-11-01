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

#include "paddle/fluid/framework/new_executor/feed_fetch_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

namespace paddle {
namespace framework {

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
          platform::errors::InvalidArgument(
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
        platform::errors::NotFound("Variable %s should not be nullptr.",
                                   feed_names[i]));
    feed_tensors.push_back(feed_var->Get<phi::DenseTensor>());
  }

  out->resize(micro_batch_num);
  if (micro_batch_num < 2) {
    (*out)[0] = std::move(feed_tensors);
    return;
  }

  for (size_t i = 0; i < feed_tensors.size(); ++i) {
    int64_t numel_size = feed_tensors[i].dims()[0];
    PADDLE_ENFORCE_EQ(numel_size % micro_batch_num,
                      0,
                      platform::errors::InvalidArgument(
                          "Split expects feed data (%s)'s dim[0] (%d) is "
                          "diviable by micro_batch_num (%d).",
                          feed_names[i],
                          numel_size,
                          micro_batch_num));
    int64_t split_size = (numel_size + micro_batch_num - 1) / micro_batch_num;
    VLOG(4) << "Split feed data:" << feed_names[i] << ", dims:("
            << feed_tensors[i].dims()
            << "), micro_batch_num:" << micro_batch_num;
    for (int64_t j = 0; j < micro_batch_num; ++j) {
      (*out)[j].resize(i + 1);
      (*out)[j][i].ShareDataWith(
          feed_tensors[i].Slice(j * split_size, j * split_size + split_size));
    }
  }
}

void FetchTensors(const std::vector<std::string>& job_fetch_names,
                  const std::vector<std::string>& fetch_var_names,
                  const int64_t micro_batch_id,
                  Scope* scope,
                  FetchUnmergedList* fetch_list) {
  PADDLE_ENFORCE_GT(
      fetch_list->size(),
      micro_batch_id,
      phi::errors::Unavailable("The fetch list size (%lld) should be greater "
                               "than micro_batch_id (%lld)",
                               fetch_list->size(),
                               micro_batch_id));

  fetch_list->at(micro_batch_id).resize(fetch_var_names.size());
  for (auto& var_name : job_fetch_names) {
    int col = find(fetch_var_names.begin(), fetch_var_names.end(), var_name) -
              fetch_var_names.begin();
    auto* var = scope->FindVar(var_name);
    auto& src = var->Get<phi::DenseTensor>();
    auto* dst =
        &(PADDLE_GET(phi::DenseTensor, fetch_list->at(micro_batch_id)[col]));
    TensorCopySync(src, platform::CPUPlace(), dst);
  }
}

void MergeFetchTensors(const FetchUnmergedList& fetch_list,
                       const int64_t micro_batch_num,
                       FetchList* out) {
  if (fetch_list.size() == 0) return;

  PADDLE_ENFORCE_EQ(
      fetch_list.size(),
      micro_batch_num,
      phi::errors::Unavailable("The fetch_list size (%lld) shoule be equal to "
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
    phi::DenseTensor merge_tensor;
    MergeLoDTensor(&merge_tensor, tensors_ptr, platform::CPUPlace());
    out->at(i) = std::move(merge_tensor);
  }
}

}  // namespace framework
}  // namespace paddle
