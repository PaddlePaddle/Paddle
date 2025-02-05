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

#include "paddle/fluid/distributed/ps/service/ps_local_client.h"
#include "paddle/fluid/distributed/ps/table/table.h"

namespace paddle::distributed {
int32_t PsLocalClient::Initialize() {
  const auto& downpour_param = _config.server_param().downpour_server_param();
  TableManager::Instance().Initialize();
  for (int i = 0; i < downpour_param.downpour_table_param_size(); ++i) {
    auto* table = CREATE_PSCORE_CLASS(
        Table, downpour_param.downpour_table_param(i).table_class());
    table->SetShard(0, 1);
    table->Initialize(downpour_param.downpour_table_param(i),
                      _config.fs_client_param());
    _table_map[downpour_param.downpour_table_param(i).table_id()].reset(table);
  }
  return 0;
}

::std::future<int32_t> PsLocalClient::Shrink(uint32_t table_id,
                                             const std::string threshold) {
  // threshold not use
  auto* table_ptr = GetTable(table_id);
  table_ptr->Shrink("");
  return done();
}

::std::future<int32_t> PsLocalClient::Load(const std::string& epoch,
                                           const std::string& mode) {
  for (auto& it : _table_map) {
    Load(it.first, epoch, mode);
  }
  return done();
}
::std::future<int32_t> PsLocalClient::Load(uint32_t table_id,
                                           const std::string& epoch,
                                           const std::string& mode) {
  auto* table_ptr = GetTable(table_id);
  table_ptr->Load(epoch, mode);
  return done();
}

::std::future<int32_t> PsLocalClient::Save(const std::string& epoch,
                                           const std::string& mode) {
  for (auto& it : _table_map) {
    Save(it.first, epoch, mode);
  }
  return done();
}
::std::future<int32_t> PsLocalClient::Save(uint32_t table_id,
                                           const std::string& epoch,
                                           const std::string& mode) {
  auto* table_ptr = GetTable(table_id);
  table_ptr->Flush();
  table_ptr->Save(epoch, mode);
  return done();
}

::std::future<int32_t> PsLocalClient::Clear() { return done(); }
::std::future<int32_t> PsLocalClient::Clear(uint32_t table_id) {
  return done();
}

::std::future<int32_t> PsLocalClient::Flush() {
  // no need
  return done();
}

::std::future<int32_t> PsLocalClient::StopServer() {
  // no need
  return done();
}

::std::future<int32_t> PsLocalClient::PullDense(Region* regions,
                                                size_t region_num,
                                                size_t table_id) {
  auto* accessor = GetTableAccessor(table_id);
  auto* table_ptr = GetTable(table_id);

  uint32_t num_per_shard =
      DenseDimPerShard(accessor->GetAccessorInfo().fea_dim, 1);

  std::vector<float> region_buffer;
  region_buffer.resize(num_per_shard);

  TableContext table_context;
  table_context.value_type = Dense;
  table_context.pull_context.values = region_buffer.data();
  table_context.num = region_buffer.size();
  table_ptr->Pull(table_context);
  //  table_ptr->PullDense(region_buffer.data(), region_buffer.size());

  size_t region_idx = 0;
  size_t region_data_idx = 0;
  size_t shard_data_size = num_per_shard;
  size_t shard_buffer_remain = shard_data_size * sizeof(float);
  PADDLE_ENFORCE_EQ(
      shard_buffer_remain,
      region_buffer.size() * sizeof(float),
      common::errors::PreconditionNotMet("pull dense size error."));
  size_t index = 0;
  while (shard_buffer_remain > 0 && region_idx < region_num) {
    auto& region = regions[region_idx];
    if (region.size - region_data_idx >= shard_buffer_remain) {
      memcpy(reinterpret_cast<void*>(region.data + region_data_idx),
             reinterpret_cast<uint8_t*>(
                 reinterpret_cast<void*>(region_buffer.data())) +
                 index,
             shard_buffer_remain);
      region_data_idx += shard_buffer_remain;
      shard_buffer_remain = 0;
    } else if (region.size - region_data_idx == 0) {
      ++region_idx;
      region_data_idx = 0;
    } else {
      memcpy(reinterpret_cast<void*>(region.data + region_data_idx),
             reinterpret_cast<uint8_t*>(
                 reinterpret_cast<void*>(region_buffer.data())) +
                 index,
             region.size - region_data_idx);
      shard_buffer_remain -= (region.size - region_data_idx);
      index += (region.size - region_data_idx);
      ++region_idx;
      region_data_idx = 0;
    }
  }

  return done();
}

::std::future<int32_t> PsLocalClient::PushDenseParam(const Region* regions,
                                                     size_t region_num,
                                                     size_t table_id) {
  auto* accessor = GetTableAccessor(table_id);
  auto* table_ptr = GetTable(table_id);

  std::vector<float> region_buffer;
  region_buffer.resize(DenseDimPerShard(accessor->GetAccessorInfo().fea_dim, 1),
                       0);
  for (size_t i = 0, offset = 0; i < region_num; ++i) {
    uint32_t data_num = regions[i].size / sizeof(float);
    memcpy(region_buffer.data() + offset, regions[i].data, regions[i].size);
    offset += data_num;
  }

  TableContext table_context;
  table_context.value_type = Dense;
  table_context.push_context.values = region_buffer.data();
  table_context.push_context.is_param = true;
  table_context.num = region_buffer.size();

  table_ptr->Push(table_context);
  // table_ptr->PushDenseParam(region_buffer.data(), region_buffer.size());

  return done();
}

::std::future<int32_t> PsLocalClient::PushDenseRawGradient(
    int table_id,
    float* total_send_data,
    size_t total_send_data_size,
    void* callback) {
  VLOG(1) << "wxx push_dense_raw_gradient";

  PSClientClosure* closure = reinterpret_cast<PSClientClosure*>(callback);

  auto* table_ptr = GetTable(table_id);

  TableContext table_context;
  table_context.value_type = Dense;
  table_context.push_context.values = total_send_data;
  table_context.num = total_send_data_size;
  //  table_ptr->PushDense(total_send_data, total_send_data_size);
  table_ptr->Push(table_context);

  delete closure;
  return done();
}

::std::future<int32_t> PsLocalClient::PushDense(const Region* regions,
                                                size_t region_num,
                                                size_t table_id) {
  auto* accessor = GetTableAccessor(table_id);
  auto* table_ptr = GetTable(table_id);

  std::vector<float> region_buffer;
  region_buffer.resize(
      DenseDimPerShard(accessor->GetAccessorInfo().fea_dim, 1));
  size_t data_size = region_buffer.size();
  for (size_t i = 0, offset = 0; i < region_num; ++i) {
    uint32_t data_num = regions[i].size / sizeof(float);
    PADDLE_ENFORCE_LE(
        offset + data_num,
        data_size,
        common::errors::PreconditionNotMet(
            "invalid dense size, cur pos[%d] data_num[%d] size[%d]",
            offset,
            data_num,
            data_size));
    memcpy(region_buffer.data() + offset, regions[i].data, regions[i].size);
    offset += data_num;
  }

  TableContext table_context;
  table_context.value_type = Dense;
  table_context.push_context.values = region_buffer.data();
  table_context.num = region_buffer.size();
  //  table_ptr->PushDense(total_send_data, total_send_data_size);
  table_ptr->Push(table_context);

  return done();
}

::std::future<int32_t> PsLocalClient::PullSparsePtr(
    int shard_id,
    char** select_values,
    size_t table_id,
    const uint64_t* keys,
    size_t num,
    uint16_t pass_id,
    const std::vector<std::unordered_map<uint64_t, uint32_t>>& keys2rank_vec,
    const uint16_t& /**dim_id*/) {
  // FIXME
  // auto timer =
  // std::make_shared<CostTimer>("pslib_downpour_client_pull_sparse");
  // auto local_timer =
  // std::make_shared<CostTimer>("pslib_downpour_client_pull_sparse_local");
  // 将key拆分到各shard请求，并记录原始对应value指针
  auto* table_ptr = GetTable(table_id);

  TableContext table_context;
  table_context.value_type = Sparse;
  table_context.pull_context.keys = keys;
  table_context.pull_context.ptr_values = select_values;
  table_context.use_ptr = true;
  table_context.num = num;
  table_context.shard_id = shard_id;
  table_context.pass_id = pass_id;

  //  table_ptr->PullSparsePtr(select_values, keys, num);
  table_ptr->Pull(table_context);

  return done();
}

::std::future<int32_t> PsLocalClient::PrintTableStat(uint32_t table_id,
                                                     uint16_t pass_id,
                                                     size_t threshold) {
  auto* table_ptr = GetTable(table_id);
  std::pair<int64_t, int64_t> ret = table_ptr->PrintTableStat();
  VLOG(0) << "table id: " << table_id << ", feasign size: " << ret.first
          << ", mf size: " << ret.second;
  // > 50亿，40%内存
  if (static_cast<size_t>(ret.first) > threshold) {
    VLOG(0) << "run cache table";
    table_ptr->CacheTable(pass_id);
  }
  return done();
}

::std::future<int32_t> PsLocalClient::SaveCacheTable(uint32_t table_id,
                                                     uint16_t pass_id,
                                                     size_t threshold) {
  auto* table_ptr = GetTable(table_id);
  std::pair<int64_t, int64_t> ret = table_ptr->PrintTableStat();
  VLOG(1) << "table id: " << table_id << ", feasign size: " << ret.first
          << ", mf size: " << ret.second;
  if (ret.first > (int64_t)threshold) {
    VLOG(1) << "run cache table";
    table_ptr->CacheTable(pass_id);
  }
  return done();
}

::std::future<int32_t> PsLocalClient::PushSparseRawGradient(
    size_t table_id,
    const uint64_t* keys,
    const float** update_values,
    size_t num,
    void* callback) {
  PSClientClosure* closure = reinterpret_cast<PSClientClosure*>(callback);
  auto* table_ptr = GetTable(table_id);

  TableContext table_context;
  table_context.value_type = Sparse;
  table_context.push_context.keys = keys;
  table_context.push_context.ptr_values = update_values;
  table_context.num = num;
  table_context.use_ptr = true;

  // table_ptr->PushSparse(keys, update_values, num);
  table_ptr->Push(table_context);
  delete closure;
  return done();
}

::std::future<int32_t> PsLocalClient::PushSparse(size_t table_id,
                                                 const uint64_t* keys,
                                                 const float** update_values,
                                                 size_t num) {
  auto* table_ptr = GetTable(table_id);

  TableContext table_context;
  table_context.value_type = Sparse;
  table_context.push_context.keys = keys;
  table_context.push_context.ptr_values = update_values;
  table_context.num = num;
  table_context.use_ptr = true;

  //  table_ptr->PushSparse(keys, update_values, num);
  table_ptr->Push(table_context);
  return done();
}

::std::future<int32_t> PsLocalClient::SetDayId(size_t table_id, int day_id) {
  auto* table_ptr = GetTable(table_id);
  table_ptr->SetDayId(day_id);
  return done();
}

}  // namespace paddle::distributed
