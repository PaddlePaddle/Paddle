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

#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/ps/table/accessor.h"

namespace paddle {
namespace distributed {

class CommMergeAccessor : public ValueAccessor {
 public:
  CommMergeAccessor() {}
  virtual ~CommMergeAccessor() {}
  virtual int Initialize();
  virtual void SetTableInfo(AccessorInfo &info);
  virtual size_t GetTableInfo(InfoKey key);
  // value维度
  size_t dim();
  // value各个维度的size
  size_t dim_size(size_t dim);
  // value各维度相加总size
  size_t size();
  // pull value维度
  size_t select_dim();
  // pull value各个维度的size
  size_t select_dim_size(size_t dim);
  // pull value各维度相加总size
  size_t select_size();
  // push value维度
  size_t update_dim();
  // push value各个维度的size
  size_t update_dim_size(size_t dim);
  // push value各维度相加总size
  size_t update_size();
  size_t fea_dim() { return _config.fea_dim(); }
  // 判断该value是否进行shrink
  virtual bool Shrink(float * /*value*/);
  // 判断该value是否在save阶段dump,
  // param作为参数用于标识save阶段，如downpour的xbox与batch_model
  virtual bool Save(float * /*value*/, int /*param*/);

  // keys不存在时，为values生成随机值
  virtual int32_t Create(float **value, size_t num);
  // 从values中选取到select_values中
  virtual int32_t Select(float **select_values, const float **values,
                         size_t num);
  // 将update_values聚合到一起
  virtual int32_t Merge(float **update_values,
                        const float **other_update_values, size_t num);
  // 将update_values聚合到一起，通过it.next判定是否进入下一个key
  // virtual int32_t Merge(float** update_values, iterator it);
  // 将update_values更新应用到values中
  virtual int32_t Update(float **values, const float **update_values,
                         size_t num);

  virtual int SetWeight(float **values, const float **update_values,
                         size_t num);
  virtual std::string ParseToString(const float *value, int param) {
    return "";
  }
  virtual int ParseFromString(const std::string &str, float *v) { return 0; }
};
}  // namespace distributed
}  // namespace paddle
