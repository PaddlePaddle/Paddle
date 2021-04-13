// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/npu_pinned_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

void NPUPinnedAllocator::ProcessEvents() {
  // 更新
}

bool NPUPinnedAllocator::IsAllocThreadSafe() const { return true; }

void NPUPinnedAllocator::FreeImpl(Allocation *allocation) {
  PADDLE_ENFORCE_EQ(
      BOOST_GET_CONST(platform::NPUPinnedPlace, allocation->place()), place_,
      platform::errors::PermissionDenied("****** TODO :NPU memory is freed in "
                                         "incorrect device. This may be a "
                                         "bug"));
  void *p = allocation->ptr();

  // TODO(liym27): 处理所有的 event。

  // std::unordered_map<void*, aclrtEvent> npu_events;
  // event = npu_events.at(p);
  // aclrtEventStatus status = ACL_EVENT_STATUS_COMPLETE;
  //
  // 获取状态
  // aclError err = aclrtQueryEvent(event, &status);

  if (/*结束的状态*/) {
    // 如果该 allocation 对应的event 已经结束了，则释放空间，否则不
    // TODO(liym27): 销毁event
    free(p);
    delete allocation;

  } else {
    VLOG << "当前不释放空间";
    // 注意：我觉得这种设计会有一个问题，就是有空间没有被释放，那这部分空间什么时候释放呢？
    // 程序退出时释放，以及下次 malloc吧。
  }

  return;
}

Allocation *NPUPinnedAllocator::AllocateImpl(size_t size) {
  // 1. TODO(liym27): 处理event，**并释放已用完的空间**
  // 2. 暂时不写这个：查看可用空间缓存中，是否有符合要求的，直接用。
  void *p;
  int error = posix_memalign(&p, kAlignment, size);

  PADDLE_ENFORCE_EQ(
      error, 0,
      platform::errors::ResourceExhausted(
          "Fail to alloc memory of %ld size, error code is %d.", size, error));
  return new Allocation(p, size, platform::NPUPinnedPlace());
}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
