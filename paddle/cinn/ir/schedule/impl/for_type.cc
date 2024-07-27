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
#include "paddle/common/enforce.h"
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
#define CINN_IR_SCHEDULE_END(err_msg_level)                                 \
  }                                                                         \
  catch (const utils::ErrorHandler& err_handler) {                          \
    PADDLE_THROW(                                                           \
        phi::errors::Fatal(err_handler.FormatErrorMessage(err_msg_level))); \
  }

void DyScheduleImpl::MutateForType(const Expr& loop,
                                   ForType for_type,
                                   int factor) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "MutateForType";
  std::ostringstream os;
  auto* for_node = loop.As<ir::For>();
  if (!for_node) {
    os << "Loop param must be For node! Please check!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }

  if (!for_node->is_serial()) {
    os << "Loop is not serial, current for loop type is "
       << static_cast<int>(for_node->for_type()) << ", and it can't become "
       << static_cast<int>(for_type) << "!\n";
  }

  auto loop_copy = ir::ir_utils::IRCopy(loop, /* copy_buffer_node = */ false);
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
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::Parallel(const Expr& loop) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Parallel";
  std::ostringstream os;
  MutateForType(loop, ForType::Parallel);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::Vectorize(const Expr& loop, int factor) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Vectorize";
  std::ostringstream os;

  if (factor <= 0) {
    os << "vectorize factor should be more than 0\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  if (!loop.As<For>()->extent.is_constant()) {
    os << "The loop to be vectorized should be constant!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  MutateForType(loop, ForType::Vectorized, factor);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::Unroll(const Expr& loop) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Unroll";
  std::ostringstream os;
  if (!loop.As<For>()->extent.is_constant()) {
    os << "The loop to be unrolled should be constant!\n";
    throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
  }
  MutateForType(loop, ForType::Unrolled);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::Bind(const Expr& loop, const std::string& thread_axis) {
  auto bindNvHygon = [&](const std::array<int, 3>& kMaxBlockDims,
                         const std::array<int, 3>& kMaxGridDims) {
    CINN_IR_SCHEDULE_BEGIN();
    std::string primitive = "Bind";
    std::ostringstream os;

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
    auto check_offset = [&](const char& c) -> bool {
      // TODO(BiynXu): rewrite the function after we have a mechanism to
      // calculate the upper bound of symbols.
      return true;
    };
    if (thread_axis[0] == 'b') {
      if (!check_offset(thread_axis[0])) {
        os << "Invalid Bind! The extent of loop is out of range on grid "
              "size!\n";
        throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
      }
      MutateForType(loop, ForType::GPUBlock, offset);
    } else {
      if (!check_offset(thread_axis[0])) {
        os << "Invalid Bind! The extent of loop is out of range on block "
              "size!\n";
        throw IRScheduleErrorHandler(primitive, os.str(), module_expr_);
      }
      MutateForType(loop, ForType::GPUThread, offset);
    }
    CINN_IR_SCHEDULE_END(this->err_msg_level_);
  };
  cinn::common::DefaultDeviceTarget().arch.Match(
      [&](std::variant<common::UnknownArch, common::X86Arch, common::ARMArch>) {
        // nothing
      },
      [&](common::NVGPUArch) {
#ifdef CINN_WITH_CUDA
        auto cur_dev_info =
            common::DevInfoMgr<common::NVGPUArch>::GetDevInfo(0);
        const std::array<int, 3> kMaxBlockDims =
            cur_dev_info->GetMaxBlockDims();
        const std::array<int, 3> kMaxGridDims = cur_dev_info->GetMaxGridDims();
        bindNvHygon(kMaxBlockDims, kMaxGridDims);
#endif
      },
      [&](common::HygonDCUArchHIP) {
#ifdef CINN_WITH_HIP
        auto cur_dev_info =
            common::DevInfoMgr<common::HygonDCUArchHIP>::GetDevInfo(0);
        const std::array<int, 3> kMaxBlockDims =
            cur_dev_info->GetMaxBlockDims();
        const std::array<int, 3> kMaxGridDims = cur_dev_info->GetMaxGridDims();
        bindNvHygon(kMaxBlockDims, kMaxGridDims);
#endif
      });
}
}  // namespace ir
}  // namespace cinn
