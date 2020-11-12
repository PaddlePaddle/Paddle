//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

// network header files
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>
#endif

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"
#include "paddle/fluid/string/string_helper.h"

// #include "paddle/fluid/framework/var_type_traits.h"

namespace paddle {
namespace imperative {

struct ParallelStrategy {
  int nranks_{1};
  int local_rank_{0};
  std::vector<std::string> trainer_endpoints_{};
  std::string current_endpoint_{""};
  int nrings_{1};
};

class ParallelContext {
 public:
  explicit ParallelContext(const ParallelStrategy& strategy,
                           const platform::Place& place)
      : strategy_(strategy), place_(place) {}

  virtual ~ParallelContext() {}

  virtual void Init() = 0;

  virtual void AllReduce(const framework::Variable& src,
                         framework::Variable* dst, int ring_id = 0,
                         bool use_calc_stream = false) = 0;
  virtual void SyncCalcStream(const platform::Place& place) = 0;
  virtual void SyncCommStream(const platform::Place& place,
                              int ring_id = 0) = 0;

  // virtual void Print_ParallelStrategy() = 0;

 protected:
  ParallelStrategy strategy_;
  platform::Place place_;
};

#if defined(PADDLE_WITH_NCCL)
class NCCLParallelContext : public ParallelContext {
 public:
  explicit NCCLParallelContext(const ParallelStrategy& strategy,
                               const platform::Place& place)
      : ParallelContext(strategy, place) {}

  // void Print_ParallelStrategy(){
  //     VLOG(0) << "init nccl context nranks: " << strategy_.nranks_
  //           << " local rank: " << strategy_.local_rank_ << " gpu id: " << 0
  //           << " ring id: " << 0;
  // }

  ~NCCLParallelContext() {}

  void BcastNCCLId(ncclUniqueId* nccl_id, int root);

  void Init() override;

  void AllReduce(const framework::Tensor& src, framework::Tensor* dst,
                 paddle::platform::NCCLComm* comm, cudaStream_t stream);
  void AllReduce(const framework::SelectedRows& src,
                 framework::SelectedRows* dst, const ParallelStrategy& strategy,
                 cudaStream_t stream, paddle::platform::NCCLComm* comm);
  void AllReduce(const framework::Variable& src, framework::Variable* dst,
                 int ring_id, bool use_calc_stream) override;

  const platform::Place& GetVarPlace(const framework::Variable& src);
  void SyncCalcStream(const platform::Place& place) override;

  void SyncCommStream(const platform::Place& place, int ring_id) override;

 protected:
  void RecvNCCLID(const std::string& endpoint, ncclUniqueId* nccl_id);

  void SendNCCLID(const std::string& endpoint, ncclUniqueId* nccl_id);
};
#endif

}  //  namespace imperative
}  //  namespace paddle
