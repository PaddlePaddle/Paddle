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

#include "paddle/fluid/distributed/table/common_dense_table.h"

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

int FLAGS_pslib_table_save_max_retry_dense = 3;

void CommonDenseTable::create_initializer(const std::string& attr,
                                          const std::string& name) {
  auto slices = string::split_string<std::string>(attr, "&");

  if (slices[0] == "gaussian_random") {
    initializers_[name] = new GaussianInitializer(slices);
  } else if (slices[0] == "fill_constant") {
    initializers_[name] = new FillConstantInitializer(slices);
  } else if (slices[0] == "uniform_random") {
    initializers_[name] = new UniformInitializer(slices);
  } else if (slices[0] == "truncated_gaussian_random") {
    initializers_[name] = new TruncatedGaussianInitializer(slices);
  } else {
    PADDLE_THROW(
        platform::errors::InvalidArgument("%s can not be supported", name));
  }
}

int32_t CommonDenseTable::initialize() {
  _shards_task_pool.resize(task_pool_size_);
  for (int i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }

  sync = _config.common().sync();
  VLOG(1) << "table " << _config.common().table_name() << " is sync: " << sync;
  _global_lr = new float(1.0);

  initialize_value();
  initialize_optimizer();
  return 0;
}

int32_t CommonDenseTable::initialize_value() {
  auto common = _config.common();
  int size = static_cast<int>(common.params().size());
  values_.resize(size);
  total_dim_ = 0;
  for (int x = 0; x < size; ++x) {
    auto& varname = common.params()[x];
    auto& dim = common.dims()[x];
    if (varname == "Param") {
      param_dim_ = dim;
      param_idx_ = x;
    }
    auto& initializer = common.initializers()[x];
    total_dim_ += dim;

    create_initializer(initializer, varname);
    values_[x].resize(dim);
    names_index_[varname] = x;

    for (int y = 0; y < dim; ++y) {
      values_[x][y] = initializers_[varname]->GetValue();
    }
  }
  VLOG(1) << "CommonDenseTable::initialize_value total dim: " << total_dim_;

  pull_reservoir_ = ReservoirValue<float>(param_dim_);
  return 0;
}

int32_t CommonDenseTable::initialize_optimizer() {
  auto common = _config.common();
  auto name = common.name();
  auto attrs = common.attributes();

  if (name == "sgd") {
    optimizer_ = std::make_shared<DSGD>(common, &values_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "adam") {
    optimizer_ = std::make_shared<DAdam>(common, &values_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "adam_d2sum"){
    optimizer_ = std::make_shared<DAdamD2Sum>(common, &values_);
    //optimizer_->set_global_lr(_global_lr);  //no use
  } else if (name == "sum") {
    optimizer_ = std::make_shared<DSUM>(common, &values_);
  } else {
    VLOG(0) << "init optimizer failed";
  }
  VLOG(3) << "init optimizer " << name << " done";
  return 0;
}

int32_t CommonDenseTable::set_global_lr(float* lr) {
  _global_lr = lr;
  optimizer_->set_global_lr(_global_lr);
  return 0;
}

int32_t CommonDenseTable::pull_dense(float* pull_values, size_t num) {
  std::copy(values_[param_idx_].begin(), values_[param_idx_].end(),
            pull_values);
  return 0;
}

int32_t CommonDenseTable::push_dense_param(const float* values, size_t num) {
  PADDLE_ENFORCE_GE(
      num, param_dim_,
      paddle::platform::errors::InvalidArgument(
          "update desne param numel expected %d, but got %d", param_dim_, num));
  std::copy_n(values, param_dim_, values_[param_idx_].begin());
  return 0;
}

int32_t CommonDenseTable::pour() {
  pull_reservoir_.avg();
  _push_dense(pull_reservoir_.values.data(), pull_reservoir_.values.size());
  pull_reservoir_.reset();
  std::cout << "zcb debug CommonDenseTable::pour() done\n";
  return 0;
}

int32_t CommonDenseTable::push_dense(const float* values, size_t num) {
  if (sync) {
    std::future<int> task =
        _shards_task_pool[0]->enqueue([this, &values]() -> int {
          pull_reservoir_.add(values, param_dim_);
          return 0;
        });
    task.wait();
  } else {
    _push_dense(values, num);
  }
  return 0;
}

int32_t CommonDenseTable::_push_dense(const float* values, size_t num) {
  PADDLE_ENFORCE_GE(
      num, param_dim_,
      paddle::platform::errors::InvalidArgument(
          "update desne numel expected %d, but got %d", param_dim_, num));

  std::vector<int> buckets = bucket(param_dim_, task_pool_size_);
  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &buckets, &values]() -> int {
          auto begin = buckets[shard_id];
          auto end = buckets[shard_id + 1];
          optimizer_->update(values, param_dim_, begin, end);
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  std::cout << "zcb debug CommonDenseTable::_push_dense done\n";
  return 0;
}


int32_t CommonDenseTable::load(const std::string& path, const std::string& param) {
  int load_param = atoi(param.c_str());
  FsChannelConfig channel_config;

  channel_config.converter = _value_accesor->converter(load_param).converter;
  channel_config.deconverter = _value_accesor->converter(load_param).deconverter;
  bool is_read_failed = false;
  int err_no = 0;
  int retry_num = 0;
  do {
    is_read_failed = false;
    try {
      size_t dim_idx = 0;
      float dim_data_buffer[2];
      std::string line_data;
      channel_config.path = paddle::string::format_string("%s/part-%03d", 
          table_dir(path).c_str(), _shard_idx);
      VLOG(0) << "CommonDenseTable::load path " << channel_config.path;
      err_no = 0;
      auto read_channel = _afs_client.open_r(channel_config, 0, &err_no);
      auto common = _config.common();
      int size = static_cast<int>(common.params().size());
      float data_buffer[2]; //actually only one float here
      float* data_buff_ptr = data_buffer;
      for (int x = 0; x < size; ++x) {
        auto& dim = common.dims()[x];
        for (int y = 0; y < dim; ++y) {
          if (read_channel->read_line(line_data) != 0) {
              break;
          }
          auto str_len = paddle::string::str_to_float(line_data.data(), data_buff_ptr);
          CHECK(str_len == 1) << "only expect 1 float: " << str_len;
          values_[x][y] = data_buffer[0];
          ++dim_idx;
          VLOG(2) << "CommonDenseTable::load x: " << x << " y: " << y << " value: " << values_[x][y];
        }
      }

      read_channel->close();

      if (err_no == -1 || dim_idx < total_dim_) {
        if (retry_num > paddle::distributed::FLAGS_pslib_table_save_max_retry_dense) {
          LOG(ERROR) << "DownpourDenseTable load failed reach max limit!";
          exit(-1);
        }
        ++retry_num;
        LOG(ERROR) << "DownpourDenseTable load failed after read , retry it! path:"
                   << channel_config.path << ", retry_num=" << retry_num;
        continue;
      }
      LOG(INFO) << "DownpourDenseTable load success, path:" << channel_config.path;
    } catch (...) {
      is_read_failed = true;
      LOG(ERROR) << "DownpourDenseTable load failed, retry it! path:" << channel_config.path;
    }
  } while (is_read_failed);
  return 0;
}

int32_t CommonDenseTable::save(const std::string& path, const std::string& param) {
  int save_param = atoi(param.c_str());
  uint32_t feasign_size;
  VLOG(0) << "CommonDenseTable::save path " << path;

  FsChannelConfig channel_config;
  if (_config.compress_in_save()) {
      channel_config.path = paddle::string::format_string("%s/part-%03d.gz", 
          table_dir(path).c_str(), _shard_idx);
  } else {
      channel_config.path = paddle::string::format_string("%s/part-%03d", 
          table_dir(path).c_str(), _shard_idx);
  }
  _afs_client.remove(channel_config.path);
  channel_config.converter = _value_accesor->converter(save_param).converter;
  channel_config.deconverter = _value_accesor->converter(save_param).deconverter;

  //float dim_data_buffer[_data.cols()];
  bool is_write_failed = false;
  std::vector<std::string> result_buffer;
  result_buffer.reserve(total_dim_);

  auto common = _config.common();
  int size = static_cast<int>(common.params().size());
  for (int x = 0; x < size; ++x) {
    auto& varname = common.params()[x];
    auto& dim = common.dims()[x];
    VLOG(0) << "CommonDenseTable::save dim " << x << " size: " << dim;
    for (int y = 0; y < dim; ++y) {
      result_buffer.emplace_back(std::move(std::to_string(values_[x][y])));
    }
  }

  int retry_num = 0;
  int err_no = 0;
  do {
      err_no = 0;
      is_write_failed = false;
      feasign_size = 0;
      // 40M
      auto write_channel = _afs_client.open_w(channel_config, 1024 * 1024 * 40, &err_no);
      for (auto& t : result_buffer) {
          if (0 != write_channel->write_line(t)) {
              ++retry_num;
              is_write_failed = true;
              LOG(ERROR) << "DownpourDenseTable save failed, retry it! "
                  "path:" << channel_config.path << ", retry_num=" << retry_num;
              break;
          }
      }
      ++feasign_size;
      write_channel->close();
      if (err_no == -1) {
          ++retry_num;
          is_write_failed = true;
          LOG(ERROR) << "DownpourDenseTable save failed after write, retry it! "
                     << "path:" << channel_config.path << ", retry_num=" << retry_num;
      }
      if (is_write_failed) {
          _afs_client.remove(channel_config.path);
      }
      if (retry_num > paddle::distributed::FLAGS_pslib_table_save_max_retry_dense) {
          LOG(ERROR) << "DownpourDenseTable save failed reach max limit!";
          exit(-1);
      }
  } while (is_write_failed);
  LOG(INFO) << "DownpourDenseTable save success, path:" << channel_config.path;
  return feasign_size;
}

}  // namespace distributed
}  // namespace paddle
