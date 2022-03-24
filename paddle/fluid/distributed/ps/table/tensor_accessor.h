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
  virtual int initialize();
  virtual void GetTableInfo(AccessorInfo &info);
  // value维度
  virtual size_t dim();
  // value各个维度的size
  virtual size_t dim_size(size_t dim);
  // value各维度相加总size
  virtual size_t size();
  // pull value维度
  virtual size_t select_dim();
  // pull value各个维度的size
  virtual size_t select_dim_size(size_t dim);
  // pull value各维度相加总size
  virtual size_t select_size();
  // push value维度
  virtual size_t update_dim();
  // push value各个维度的size
  virtual size_t update_dim_size(size_t dim);
  // push value各维度相加总size
  virtual size_t update_size();
  // 判断该value是否进行shrink
  virtual bool shrink(float * /*value*/);
  // 判断该value是否在save阶段dump,
  // param作为参数用于标识save阶段，如downpour的xbox与batch_model
  virtual bool save(float * /*value*/, int /*param*/);

  // keys不存在时，为values生成随机值
  virtual int32_t create(float **value, size_t num);
  // 从values中选取到select_values中
  virtual int32_t select(float **select_values, const float **values,
                         size_t num);
  // 将update_values聚合到一起
  virtual int32_t merge(float **update_values,
                        const float **other_update_values, size_t num);
  // 将update_values聚合到一起，通过it.next判定是否进入下一个key
  // virtual int32_t merge(float** update_values, iterator it);
  // 将update_values更新应用到values中
  virtual int32_t update(float **values, const float **update_values,
                         size_t num);

  virtual int set_weight(float **values, const float **update_values,
                         size_t num);
  virtual std::string parse_to_string(const float *value, int param) {
    return "";
  }
  virtual int parse_from_string(const std::string &str, float *v) { return 0; }
};
}  // namespace distributed
}  // namespace paddle
