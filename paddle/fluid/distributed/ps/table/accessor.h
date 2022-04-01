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
#include <unordered_map>
#include <vector>
#include "paddle/fluid/distributed/common/afs_warpper.h"
#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/ps.pb.h"

namespace paddle {
namespace distributed {

struct Region {
  Region() : data(NULL), size(0) {}
  Region(char* data, size_t data_num) : data(data), size(data_num) {}
  Region(float* data, size_t data_num)
      : data(reinterpret_cast<char*>(data)), size(data_num << 2) {}
  Region(int16_t* data, size_t data_num)
      : data(reinterpret_cast<char*>(data)), size(data_num << 1) {}
  Region(int32_t* data, size_t data_num)
      : data(reinterpret_cast<char*>(data)), size(data_num << 2) {}
  Region(int64_t* data, size_t data_num)
      : data(reinterpret_cast<char*>(data)), size(data_num << 3) {}
  char* data;
  size_t size;
};

struct DataConverter {
  int param;
  std::string converter;
  std::string deconverter;
};

struct AccessorInfo {
  size_t dim;
  size_t size;
  size_t select_size;
  size_t select_dim;
  size_t update_size;
  size_t update_dim;
  size_t mf_size;
  size_t fea_dim;
};

enum InfoKey {
  DIM = 0,
  SIZE = 1,
  SELECT_SIZE = 2,
  SELECT_DIM = 3,
  UPDATE_SIZE = 4,
  UPDATE_DIM = 5,
  MF_SIZE = 6,
  FEA_DIM = 7
};

class ValueAccessor {
 public:
  ValueAccessor() {}
  virtual ~ValueAccessor() {}

  virtual int configure(const TableAccessorParameter& parameter) {
    _config = parameter;
    // data_convert结构体初始化
    if (_config.table_accessor_save_param_size() != 0) {
      for (int i = 0; i < _config.table_accessor_save_param_size(); ++i) {
        int param = _config.table_accessor_save_param(i).param();
        std::string converter =
            _config.table_accessor_save_param(i).converter();
        std::string deconverter =
            _config.table_accessor_save_param(i).deconverter();
        _data_coverter_map[param] = std::make_shared<DataConverter>();
        *(_data_coverter_map[param]) = {param, converter, deconverter};
      }
    }
    return 0;
  }
  virtual int initialize() = 0;

  virtual void SetTableInfo(AccessorInfo& info) = 0;
  virtual size_t GetTableInfo(InfoKey key) = 0;

  // value维度
  virtual size_t dim() = 0;
  // value各个维度的size
  virtual size_t dim_size(size_t dim) = 0;
  // value各维度相加总size
  virtual size_t size() = 0;

  // value中mf动态长度部分总size大小, sparse下生效
  virtual size_t mf_size() { return 0; }
  virtual bool need_extend_mf(float* value) { return false; }
  virtual bool has_mf(size_t size) { return false; }
  // pull value维度
  virtual size_t select_dim() = 0;
  // pull value各个维度的size
  virtual size_t select_dim_size(size_t dim) = 0;
  // pull value各维度相加总size
  virtual size_t select_size() = 0;
  // push value维度
  virtual size_t update_dim() = 0;
  // push value各个维度的size
  virtual size_t update_dim_size(size_t dim) = 0;
  // push value各维度相加总size
  virtual size_t update_size() = 0;
  // fea total for dense
  virtual size_t fea_dim() { return _config.fea_dim(); }
  // converter for save
  virtual std::string get_converter(int param) {
    auto itr = _data_coverter_map.find(param);
    if (itr == _data_coverter_map.end()) {
      return "";
    } else {
      return (*itr).second->converter;
    }
  }
  // deconverter for load
  virtual std::string get_deconverter(int param) {
    auto itr = _data_coverter_map.find(param);
    if (itr == _data_coverter_map.end()) {
      return "";
    } else {
      return (*itr).second->deconverter;
    }
  }
  // 判断该value是否进行shrink
  virtual bool shrink(float* value) = 0;

  // 判断该value是否在save阶段dump,
  // param作为参数用于标识save阶段，如downpour的xbox与batch_model
  virtual bool save(float* value, int param) = 0;
  // update delta_score and unseen_days after save
  virtual void update_stat_after_save(float* value, int param) {}

  // keys不存在时，为values生成随机值
  virtual int32_t create(float** value, size_t num) = 0;
  virtual bool create_value(int type, const float* value) { return true; }
  // 从values中选取到select_values中
  virtual int32_t select(float** select_values, const float** values,
                         size_t num) = 0;
  // 将update_values聚合到一起
  virtual int32_t merge(float** update_values,
                        const float** other_update_values, size_t num) = 0;
  // 将update_values聚合到一起，通过it.next判定是否进入下一个key
  // virtual int32_t merge(float** update_values, iterator it);
  // 将update_values更新应用到values中
  virtual int32_t update(float** values, const float** update_values,
                         size_t num) = 0;

  // used to save model, will filter feature
  virtual std::string parse_to_string(const float* value, int param) = 0;
  //  parse value from string, used to load model
  virtual int32_t parse_from_string(const std::string& data, float* value) = 0;

  virtual FsDataConverter converter(int param) {
    FsDataConverter data_convert;
    data_convert.converter = this->get_converter(param);
    data_convert.deconverter = this->get_deconverter(param);
    return data_convert;
  }

  virtual int set_weight(float** values, const float** update_values,
                         size_t num) {
    return 0;
  }

  virtual float get_field(float* value, const std::string& name) { return 0.0; }
#define DEFINE_GET_INDEX(class, field) \
  virtual int get_##field##_index() override { return class ::field##_index(); }

 protected:
  size_t _value_size;
  size_t _select_value_size;
  size_t _update_value_size;
  TableAccessorParameter _config;
  std::unordered_map<int, std::shared_ptr<struct DataConverter>>
      _data_coverter_map;
  AccessorInfo _accessor_info;
};
REGISTER_PSCORE_REGISTERER(ValueAccessor);
}  // namespace distributed
}  // namespace paddle
