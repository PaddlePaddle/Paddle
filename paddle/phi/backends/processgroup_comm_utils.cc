// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/phi/backends/c_comm_lib.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#endif
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
#include "paddle/fluid/distributed/collective/ProcessGroupCustom.h"
#endif

namespace phi {
namespace detail {

// FIXME(paddle-dev): Since the singleton of ProcessGroup in fluid is used in
// SyncBN, the fluid symbol will be dependent on external hardware access.
// Here, the part that depends on the fluid symbol is individually encapsulated
// as a temporary function to isolate external symbol dependencies.
// In the future, the dependence on the singleton in fluid in SyncBN needs
// to be removed.
// In principle, the PHI Kernel cannot use the global singleton internally,
// and the required members need to be passed in from the eucalyptus tree.
ccl::CCLComm GetCCLComm(const Place& place, int global_gid) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_CUSTOM_DEVICE)
  paddle::distributed::ProcessGroup* pg = nullptr;
  if (paddle::distributed::ProcessGroupMapFromGid::getInstance()->has(
          global_gid)) {
    pg = paddle::distributed::ProcessGroupMapFromGid::getInstance()->get(
        global_gid);
  } else {
    return nullptr;
  }
#endif
  if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    return static_cast<paddle::distributed::ProcessGroupNCCL*>(pg)->NCCLComm(
        place);
#else
    return nullptr;
#endif
  } else if (paddle::platform::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
    return static_cast<paddle::distributed::ProcessGroupCustom*>(pg)
        ->CustomCCLComm(place);
#else
    return nullptr;
#endif
  } else {
    return nullptr;
  }
}

}  // namespace detail
}  // namespace phi
