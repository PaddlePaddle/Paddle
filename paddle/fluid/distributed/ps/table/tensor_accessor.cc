// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/ps/table/tensor_accessor.h"
#include "Eigen/Dense"

namespace paddle {
namespace distributed {

int CommMergeAccessor::initialize() { return 0; }

void CommMergeAccessor::GetTableInfo(AccessorInfo &info) {
  info.dim = dim();
  info.size = size();
  info.select_dim = select_dim();
  info.select_size = select_size();
  info.update_dim = update_dim();
  info.update_size = update_size();
  info.fea_dim = fea_dim();
}

// value 维度
size_t CommMergeAccessor::dim() { return 0; }

// value 各个维度的size
size_t CommMergeAccessor::dim_size(size_t dim) { return 0; }

// value 各维度相加总size
size_t CommMergeAccessor::size() { return 0; }

// pull value 维度
size_t CommMergeAccessor::select_dim() { return _config.embedx_dim(); }

// pull value 各个维度的size
size_t CommMergeAccessor::select_dim_size(size_t dim) { return sizeof(float); }

// pull value 各维度相加总size
size_t CommMergeAccessor::select_size() { return select_dim() * sizeof(float); }

// push value 维度
size_t CommMergeAccessor::update_dim() { return _config.embedx_dim(); }

// push value 各个维度的size
size_t CommMergeAccessor::update_dim_size(size_t dim) { return sizeof(float); }

// push value 各维度相加总size
size_t CommMergeAccessor::update_size() { return update_dim() * sizeof(float); }

// 判断该value 是否进行shrink
bool CommMergeAccessor::shrink(float * /*value*/) { return false; }

// 判断该value 是否在save阶段dump,
// param作为参数用于标识save阶段，如downpour的xbox与batch_model
bool CommMergeAccessor::save(float * /*value*/, int /*param*/) { return true; }

// keys不存在时，为values生成随机值
int32_t CommMergeAccessor::create(float **value, size_t num) { return 0; }

// 从values中选取到select_values中
int32_t CommMergeAccessor::select(float **select_values, const float **values,
                                  size_t num) {
  return 0;
}

// 将update_values聚合到一起
int32_t CommMergeAccessor::merge(float **update_values,
                                 const float **other_update_values,
                                 size_t num) {
  Eigen::Map<Eigen::MatrixXf> u_mat(update_values[0], 1, num);
  Eigen::Map<const Eigen::MatrixXf> o_mat(other_update_values[0], 1, num);
  u_mat += o_mat;
  return 0;
}

// 将update_values聚合到一起，通过it.next判定是否进入下一个key
//  int32_t merge(float** update_values, iterator it);
// 将update_values更新应用到values中
int32_t CommMergeAccessor::update(float **values, const float **update_values,
                                  size_t num) {
  return 0;
}

int CommMergeAccessor::set_weight(float **values, const float **update_values,
                                  size_t num) {
  return 0;
}

}  // namespace distributed
}  // namespace paddle
