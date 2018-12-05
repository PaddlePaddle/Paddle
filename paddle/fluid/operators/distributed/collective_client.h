// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <condition_variable>  // NOLINT
#include <sstream>
#include <string>
#include <vector>
#include "gflags/gflags.h"

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

DECLARE_int32(rpc_deadline);

namespace paddle {
namespace operators {
namespace distributed {

inline std::string GetSelectedRowsInfo(const framework::SelectedRows& slr) {
  std::stringstream ss;
  ss << "height:" << slr.height() << ", rows:[";
  for (unsigned int i = 0; i < slr.rows().size(); i++) {
    if (i != slr.rows().size() - 1) {
      ss << slr.rows()[i] << ",";
    } else {
      ss << slr.rows()[i];
    }
  }
  ss << "], dims:" << slr.value().dims();

  return ss.str();
}

class CollectiveClient {
 public:
  CollectiveClient() {}
  virtual ~CollectiveClient() {}

  // note this function will retain the rank order.
  // TODO(gongwb): Implement ringbased gather.
  // https://github.com/baidu-research/baidu-allreduce
  template <typename T>
  static void ReduceSelectedRows(const std::vector<std::string>& endpoints,
                                 const std::string& var_name,
                                 framework::Scope* local_scope,
                                 const std::string& dst_var_name,
                                 int64_t time_out = FLAGS_rpc_deadline);

  static void BroadCast(const std::vector<std::string>& endpoints,
                        const platform::DeviceContext& dev_ctx,
                        framework::Scope* scope, const std::string& var_name,
                        int64_t time_out = FLAGS_rpc_deadline);
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

#include "paddle/fluid/operators/distributed/collective_client_impl.h"
