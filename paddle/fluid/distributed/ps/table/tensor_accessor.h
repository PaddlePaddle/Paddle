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
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"

namespace paddle {
namespace distributed {

class CommMergeAccessor : public ValueAccessor {
 public:
  CommMergeAccessor() {}
  virtual ~CommMergeAccessor() {}
  virtual int Initialize();
  // 初始化AccessorInfo
  virtual void InitAccessorInfo();
  // 判断该value是否进行shrink
  virtual bool Shrink(float * /*value*/);
  // 判断该value是否在save阶段dump,
  // param作为参数用于标识save阶段，如downpour的xbox与batch_model
  virtual bool Save(float * /*value*/, int /*param*/);

  bool SaveCache(float *value, int param, double global_cache_threshold) {
    return false;
  }

  bool SaveSSD(float *value) { return false; }

  // keys不存在时，为values生成随机值
  virtual int32_t Create(float **value, size_t num);
  // 从values中选取到select_values中
  virtual int32_t Select(float **select_values,
                         const float **values,
                         size_t num);
  // 将update_values聚合到一起
  virtual int32_t Merge(float **update_values,
                        const float **other_update_values,
                        size_t num);
  // 将update_values聚合到一起，通过it.next判定是否进入下一个key
  // virtual int32_t Merge(float** update_values, iterator it);
  // 将update_values更新应用到values中
  virtual int32_t Update(float **values,
                         const float **update_values,
                         size_t num);

  virtual int SetWeight(float **values,
                        const float **update_values,
                        size_t num);
  virtual std::string ParseToString(const float *value, int param) {
    return "";
  }
  virtual int ParseFromString(const std::string &str, float *v) { return 0; }
};
}  // namespace distributed
}  // namespace paddle
