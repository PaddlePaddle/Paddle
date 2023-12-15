// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/dev_info_manager.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/ir/schedule/impl/ir_schedule.h"

namespace cinn {
namespace ir {

/** \brief A macro that guards the beginning of each implementation of schedule
 */
#define CINN_IR_SCHEDULE_BEGIN() try {
/**
 * \brief A macro that pairs with `CINN_IR_SCHEDULE_BEGIN`, handling potential
 * errors and error message printing.
 * @param primitive A string representing the kind of schedule primitive.
 * @param err_msg_level A ScheduleErrorMessageLevel enum, level of error message
 * printing
 */
#define CINN_IR_SCHEDULE_END(err_msg_level)                    \
  }                                                            \
  catch (const utils::ErrorHandler& err_hanlder) {             \
    CINN_THROW(err_hanlder.FormatErrorMessage(err_msg_level)); \
  }

void DyScheduleImpl::MutateForType(const Expr& loop,
                                   ForType for_type,
                                   int factor) {
  auto* for_node = loop.As<ir::For>();
  CHECK(for_node) << "loop param must be For node! Please check.";
  CHECK(for_node->is_serial())
      << "loop is not serial, current forloop type is "
      << static_cast<int>(for_node->for_type()) << ", and it cannot become "
      << static_cast<int>(for_type);
  auto loop_copy = ir::ir_utils::IRCopy(loop);
  auto* new_for_node = loop_copy.As<ir::For>();
  CHECK(new_for_node);
  new_for_node->set_for_type(for_type);
  if (new_for_node->is_vectorized()) {
    VectorizeInfo vec_info(0, factor);
    new_for_node->set_vectorize_info(vec_info);
  } else if (new_for_node->is_binded()) {
    BindInfo bind_info(for_type, factor, DeviceAPI::GPU);
    new_for_node->set_bind_info(bind_info);
  }
  this->Replace(loop, loop_copy);
}

void DyScheduleImpl::Parallel(const Expr& loop) {
  MutateForType(loop, ForType::Parallel);
}

void DyScheduleImpl::Vectorize(const Expr& loop, int factor) {
  CHECK_GT(factor, 0) << "vectorize factor should be more than 0";
  CHECK(loop.As<For>()->extent.is_constant())
      << "The loop to be vectorized should be constant!\n";
  MutateForType(loop, ForType::Vectorized, factor);
}

void DyScheduleImpl::Unroll(const Expr& loop) {
  CHECK(loop.As<For>()->extent.is_constant())
      << "The loop to be unrolled should be constant!\n";
  MutateForType(loop, ForType::Unrolled);
}

void DyScheduleImpl::Bind(const Expr& loop, const std::string& thread_axis) {
#ifdef CINN_WITH_CUDA
  std::string primitive = "Bind";
  std::ostringstream os;

  if (!loop.As<For>()->extent.is_constant()) {
    os << "The extent of loop to be binded should be constant!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  static std::set<std::string> thread_axes = {"blockIdx.x",
                                              "blockIdx.y",
                                              "blockIdx.z",
                                              "threadIdx.x",
                                              "threadIdx.y",
                                              "threadIdx.z"};
  if (!thread_axes.count(thread_axis)) {
    os << "The thread_axis which is " << thread_axis << " is not supported\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  int offset = thread_axis.back() - 'x';
  auto cur_dev_info =
      common::DevInfoMgr<common::Target::Arch::NVGPU>::GetDevInfo(0);
  const std::array<int, 3> kMaxBlockDims = cur_dev_info->GetMaxBlockDims();
  const std::array<int, 3> kMaxGridDims = cur_dev_info->GetMaxGridDims();
  auto check_offset = [&](const char& c) -> bool {
    auto extent = loop.As<ir::For>()->extent.as_int32();
    return extent <= (c == 'b' ? kMaxGridDims[offset] : kMaxBlockDims[offset]);
  };
  if (thread_axis[0] == 'b') {
    if (!check_offset(thread_axis[0])) {
      os << "Invalid Bind! The extent of loop is out of range on grid size!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }
    MutateForType(loop, ForType::GPUBlock, offset);
  } else {
    if (!check_offset(thread_axis[0])) {
      os << "Invalid Bind! The extent of loop is out of range on block size!\n";
      throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
    }
    MutateForType(loop, ForType::GPUThread, offset);
  }
#endif
}
}  // namespace ir
}  // namespace cinn

namespace cinn {
namespace ir {
void StScheduleImpl::MutateForType(const Expr& loop,
                                   ForType for_type,
                                   int factor) {
  auto* for_node = loop.As<ir::For>();
  CHECK(for_node) << "loop param must be For node! Please check.";
  CHECK(for_node->is_serial())
      << "loop is not serial, current forloop type is "
      << static_cast<int>(for_node->for_type()) << ", and it cannot become "
      << static_cast<int>(for_type);
  auto loop_copy = ir::ir_utils::IRCopy(loop);
  auto* new_for_node = loop_copy.As<ir::For>();
  CHECK(new_for_node);
  new_for_node->set_for_type(for_type);
  if (new_for_node->is_vectorized()) {
    VectorizeInfo vec_info(0, factor);
    new_for_node->set_vectorize_info(vec_info);
  } else if (new_for_node->is_binded()) {
    BindInfo bind_info(for_type, factor, DeviceAPI::GPU);
    new_for_node->set_bind_info(bind_info);
  }
  this->Replace(loop, loop_copy);
}

void StScheduleImpl::Parallel(const Expr& loop) {
  MutateForType(loop, ForType::Parallel);
}

void StScheduleImpl::Vectorize(const Expr& loop, int factor) {
  CHECK_GT(factor, 0) << "vectorize factor should be more than 0";
  MutateForType(loop, ForType::Vectorized, factor);
}

void StScheduleImpl::Unroll(const Expr& loop) {
  MutateForType(loop, ForType::Unrolled);
}

void StScheduleImpl::Bind(const Expr& loop, const std::string& thread_axis) {
#ifdef CINN_WITH_CUDA
  CINN_IR_SCHEDULE_BEGIN();
  static std::set<std::string> thread_axes = {"blockIdx.x",
                                              "blockIdx.y",
                                              "blockIdx.z",
                                              "threadIdx.x",
                                              "threadIdx.y",
                                              "threadIdx.z"};
  CHECK(thread_axes.count(thread_axis))
      << "thread_axis " << thread_axis << " is not supported";
  int offset = thread_axis.back() - 'x';
  auto cur_dev_info =
      cinn::common::DevInfoMgr<cinn::common::Target::Arch::NVGPU>::GetDevInfo(
          0);
  const std::array<int, 3> kMaxBlockDims = cur_dev_info->GetMaxBlockDims();
  const std::array<int, 3> kMaxGridDims = cur_dev_info->GetMaxGridDims();
  auto check_offset = [&](const char& c) -> bool {
    auto extent = loop.As<ir::For>()->extent.as_int32();
    return extent <= (c == 'b' ? kMaxGridDims[offset] : kMaxBlockDims[offset]);
  };
  if (thread_axis[0] == 'b') {
    CHECK(check_offset(thread_axis[0]))
        << "Invalid Bind! The extent of loop is out of range on grid size!\n";
    MutateForType(loop, ForType::GPUBlock, offset);
  } else {
    CHECK(check_offset(thread_axis[0]))
        << "Invalid Bind! The extent of loop is out of range on block size!\n";
    MutateForType(loop, ForType::GPUThread, offset);
  }
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
#endif
}

}  // namespace ir
}  // namespace cinn
