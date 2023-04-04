/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/lib/tensor_copy.h"

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace experimental {
static int64_t op_id = 0;
struct timeval t1;
struct timeval t2;
static std::stringstream debug_start_stream;

void XPUPaddleOpTimeTik() {
  if (std::getenv("XPU_PADDLE_OP_TIME") != nullptr) {
    gettimeofday(&t1, NULL);
  }
}

void XPUPaddleOpTimeTok(const std::string& op_name,
                        const phi::DeviceContext* dev_ctx,
                        const Backend& dev_place,
                        const DataType& kernel_data_type) {
  // 耗时统计逻辑
  if (std::getenv("XPU_PADDLE_OP_TIME") != nullptr) {
    if (platform::is_xpu_place(phi::TransToPhiPlace(dev_place))) {
      dev_ctx->Wait();
    }
    gettimeofday(&t2, NULL);
    uint32_t diff = 1000000 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec;
    std::cout << "op_name " << phi::TransToFluidOpName(op_name) << " " << diff
              << " " << dev_place << " " << kernel_data_type << std::endl;
  }
}

int64_t OpId() { return op_id; }
int64_t OpIdAdd() { return op_id++; }

phi::Place& xpu_debug_run_dev2() {
  static phi::Place dev2 = phi::CPUPlace();
  static bool inited = false;
  static std::string device = "CPU";
  if (!inited) {
    if (std::getenv("XPU_PADDLE_DEBUG_RUN_DEV2") != nullptr) {
      std::string ops(std::getenv("XPU_PADDLE_DEBUG_RUN_DEV2"));
      if (ops == "1" || ops == "XPU" || ops == "xpu") {
        dev2 = phi::XPUPlace();
        device = "XPU";
      }
    }
    inited = true;
    VLOG(3) << "XPU Paddle Debug Run Dev2: " << device;
  }
  return dev2;
}

double get_debug_thres() {
  static double thres = -10;
  static bool inited = false;
  if (!inited) {
    if (std::getenv("XPU_PADDLE_DEBUG_THRES") != nullptr) {
      std::string thres_str(std::getenv("XPU_PADDLE_DEBUG_THRES"));
      std::istringstream isthres_str(thres_str);
      isthres_str >> thres;
    }
    inited = true;
    VLOG(3) << "XPU Paddle Debug Check Threshold: " << thres;
  }
  return thres;
}

bool ContinueOrNot(const std::string& op_name) {
  auto fluid_name = phi::TransToFluidOpName(op_name);
  bool continue_or_not =
      !phi::backends::xpu::is_in_xpu_debug_black_list(fluid_name) &&
      !phi::backends::xpu::is_in_xpu_debug_black_id_list(std::to_string(op_id));
  continue_or_not =
      continue_or_not &&
      (phi::backends::xpu::is_in_xpu_debug_white_list(fluid_name) ||
       std::getenv("XPU_PADDLE_DEBUG_WHITE_LIST") == nullptr);
  continue_or_not = continue_or_not &&
                    (phi::backends::xpu::is_in_xpu_debug_white_id_list(
                         std::to_string(op_id)) ||
                     std::getenv("XPU_PADDLE_DEBUG_WHITE_ID_LIST") == nullptr);
  return continue_or_not;
}

bool ContinueRunDev2OrNot(const std::string& op_name) {
  auto fluid_name = phi::TransToFluidOpName(op_name);
  bool continue_or_not =
      !phi::backends::xpu::is_in_xpu_debug_run_dev2_black_list(fluid_name);
  return continue_or_not;
}

bool DebugOrNot() {
  bool continue_or_not = (std::getenv("XPU_PADDLE_DEBUG_GLOBAL") != nullptr ||
                          std::getenv("XPU_PADDLE_DEBUG_OP") != nullptr);
  return continue_or_not;
}

std::string GetDebugStartStr() { return debug_start_stream.str(); }

void SetDebugStartStr(const std::string& str) {
  debug_start_stream.str(std::string());
  debug_start_stream.clear();
  debug_start_stream << str;
}

std::string XPUDebugStartString(const std::string& op_name,
                                const Backend& dev_place,
                                const DataType& kernel_data_type) {
  if (ContinueOrNot(op_name)) {
    std::stringstream print_buffer;
    print_buffer << "op_name_debug " << phi::TransToFluidOpName(op_name) << " "
                 << op_id << " " << kernel_data_type << " " << dev_place << " ";
    //  << dev_place << " "
    //  << kernel_data_type << " in: ";
    return print_buffer.str();
  } else {
    return "";
  }
}

static std::pair<std::string, bool> XPUDebugStringImpl(
    const std::string& tensor_name, const Tensor& a, const Tensor& b) {
  std::stringstream print_buffer;
  bool thes_open = false;
  print_buffer << tensor_name << "-";
  if (a.name() == "") {
    print_buffer << "None-";
  } else {
    print_buffer << a.name() << "-";
  }
  // VLOG(10) << print_buffer.str();
  if (a.defined()) {
    if (a.initialized()) {
      // paddle::platform::DeviceContextPool& pool =
      //     paddle::platform::DeviceContextPool::Instance();
      // if (a.place().GetType() == AllocationType::XPU) {
      //   auto& dev_ctx = *pool.Get(a.place());
      //   dev_ctx.Wait();
      // }
      // if (b.place().GetType() == AllocationType::XPU) {
      //   auto& dev_ctx = *pool.Get(b.place());
      //   dev_ctx.Wait();
      // }
      VLOG(11) << "a " << a.dtype() << " " << a.place();
      VLOG(11) << "b " << b.dtype() << " " << b.place();
      double rmse = check_mse(a, b);
      if (rmse > get_debug_thres()) {
        thes_open = true;
      }
      print_buffer << a.dtype() << "-" << a.place().DebugString() << "-"
                   << b.place().DebugString() << "-[" << rmse << "] ";

    } else {
      print_buffer << "NOT_INITED ";
    }
  } else {
    print_buffer << "NOT_INITED_VAR ";
  }
  // VLOG(10) << print_buffer.str();
  return std::make_pair(print_buffer.str(), thes_open);
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const Tensor& a,
                           const Tensor& b) {
  if (ContinueOrNot(op_name)) {
    auto tmp = XPUDebugStringImpl(tensor_name, a, b);
    // std::string tmp = XPUDebugStringImpl(tensor_name, a, b);
    VLOG(10) << tmp.first;
    if (tmp.second) {
      return tmp.first;
    } else {
      return "";
    }
  } else {
    return "";
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const optional<Tensor>& a,
                           const optional<Tensor>& b) {
  if (a) {
    auto tmp = XPUDebugString(op_name, tensor_name, *a, *b);
    return tmp;
  } else {
    VLOG(10) << tensor_name << "-NOT_INITED_VAR ";
    return "";
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const std::vector<Tensor>& a,
                           const std::vector<Tensor>& b) {
  std::string tensors_str = "vector:{";
  bool return_or_not = false;
  if (ContinueOrNot(op_name)) {
    for (size_t i = 0; i < a.size(); i++) {
      auto tmp = XPUDebugString(op_name, tensor_name, a[i], b[i]);
      if (tmp != "") {
        return_or_not = true;
      }
      tensors_str += tmp;
    }
    std::string tmp = tensors_str + "} ";
    VLOG(10) << tmp;
    if (return_or_not) {
      return tmp;
    } else {
      return "";
    }
  } else {
    return "";
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const std::vector<Tensor*>& a,
                           const std::vector<Tensor*>& b) {
  std::string tensors_str = "vector:{";
  bool return_or_not = false;
  if (ContinueOrNot(op_name)) {
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i]) {
        auto tmp = XPUDebugString(op_name, tensor_name, *(a[i]), *(b[i]));
        if (tmp != "") {
          return_or_not = true;
        }
        tensors_str += tmp;
      }
    }
    std::string tmp = tensors_str + "} ";
    VLOG(10) << tmp;
    if (return_or_not) {
      return tmp;
    } else {
      return "";
    }
  } else {
    return "";
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const optional<std::vector<Tensor>>& a,
                           const optional<std::vector<Tensor>>& b) {
  // return XPUDebugString(op_name, tensor_name, *a, *b);
  if (a) {
    return XPUDebugString(op_name, tensor_name, *a, *b);
  } else {
    // return tensor_name + "-NOT_INITED_VAR ";
    VLOG(10) << tensor_name << "-NOT_INITED_VAR ";
    return "";
  }
}

void copy(const Tensor& src, const Place& place, bool blocking, Tensor* dst) {
  VLOG(10) << "src.name(): " << src.name();
  VLOG(10) << "dst == nullptr? " << (dst == nullptr);
  dst->set_name(src.name());
  if (!(src.impl())) {
    VLOG(10) << "src.impl() == nullptr? " << (src.impl() == nullptr);
    VLOG(10) << "dst->impl() == nullptr? " << (dst->impl() == nullptr);
    return;
  }
  auto kernel_key_set = ParseKernelKeyByInputArgs(src);
  kernel_key_set.backend_set =
      kernel_key_set.backend_set | BackendSet(phi::TransToPhiBackend(place));
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  auto target_place = phi::TransToPhiPlace(kernel_key.backend());
  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetMutable(
      target_place.GetType() == place.GetType() ? place : target_place);

  auto dense_x = TensorToDenseTensor(src);

  auto kernel_out = SetKernelOutput(dst);
  phi::MetaTensor meta_out(kernel_out);
  phi::UnchangedInferMeta(*dense_x, &meta_out);

  VLOG(6) << "start copy. ";

  if (dense_x->initialized()) {
    phi::Copy(*dev_ctx, *dense_x, place, blocking, kernel_out);
  } else {
    if (dense_x->IsInitialized()) {
      if (place.GetType() == AllocationType::CPU) {
        // dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
        // dev_ctx->HostAlloc(kernel_out, kernel_out->dtype(), 0, true);
        dev_ctx->HostAlloc(kernel_out, kernel_out->dtype());
      } else if (place.GetType() == AllocationType::XPU) {
        // dev_ctx->Alloc(kernel_out, kernel_out->dtype(), 0, false, true);
        dev_ctx->Alloc(kernel_out, kernel_out->dtype());
      }
    }
  }

  VLOG(6) << "copy finished. ";

  VLOG(10) << "dense_x->initialized(): " << dense_x->initialized();
  VLOG(10) << "dense_x->holder_or_not(): " << dense_x->IsInitialized();
  if (dense_x->IsInitialized()) {
    VLOG(10) << "dense_x->holder_ptr_or_not(): "
             << dense_x->holder_ptr_or_not();
    VLOG(10) << "dense_x->place(): " << dense_x->place();
  }
  VLOG(10) << "dense_x->meta(): " << dense_x->meta();

  VLOG(10) << "dev_ctx_place "
           << (target_place.GetType() == place.GetType() ? target_place : place)
           << std::endl;

  VLOG(10) << "kernel_out->initialized(): " << kernel_out->initialized();
  VLOG(10) << "kernel_out->holder_or_not(): " << kernel_out->IsInitialized();
  if (kernel_out->IsInitialized()) {
    VLOG(10) << "kernel_out->holder_ptr_or_not(): "
             << kernel_out->holder_ptr_or_not();
    VLOG(10) << "kernel_out->place(): " << kernel_out->place();
  }
  VLOG(10) << "kernel_out->meta(): " << kernel_out->meta();
}

void copy(const std::vector<Tensor>& src,
          const Place& place,
          bool blocking,
          std::vector<Tensor>* dst) {
  // std::vector<paddle::experimental::Tensor> result;
  dst->reserve(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    auto src_tensor = src[i];
    Tensor dst_tensor;
    copy(src_tensor, place, blocking, &dst_tensor);
    dst->emplace_back(dst_tensor);
  }
}

void copy(const std::vector<Tensor*>& src,
          const Place& place,
          bool blocking,
          std::vector<Tensor*> dst) {
  // std::vector<paddle::experimental::Tensor> result;
  // dst->reserve(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    auto src_tensor = src[i];
    if (src_tensor == nullptr) {
      // dst->emplace_back(nullptr);
      continue;
    }
    // Tensor dst_tensor;
    auto dst_tensor = dst[i];
    copy(*src_tensor, place, blocking, dst_tensor);
    // dst->emplace_back(&dst_tensor);
  }
}

// void copy(const std::vector<Tensor*> src,
//           const Place& place,
//           bool blocking,
//           std::vector<Tensor*> dst) {
//   // std::vector<paddle::experimental::Tensor> result;
//   for (size_t i = 0; i < src.size(); i++) {
//     auto src_tensor = *src[i];
//     paddle::experimental::Tensor dst_tensor;
//     copy(src_tensor, place, blocking, &dst_tensor);
//     dst.emplace_back(&dst_tensor);
//   }
// }

void copy(const paddle::optional<Tensor>& src,
          const Place& place,
          bool blocking,
          paddle::optional<Tensor>* dst) {
  // in ? copy(*src, place, blocking, &(*dst)) : dst.destroy();
  if (src) {
    copy(*src, place, blocking, &(*(*dst)));
  } else {
    dst->reset();
  }
}

void copy(const optional<std::vector<Tensor>>& src,
          const Place& place,
          bool blocking,
          optional<std::vector<Tensor>>* dst) {
  if (src) {
    copy(*src, place, blocking, &(*(*dst)));
  } else {
    dst->reset();
  }
}

std::shared_ptr<Tensor> copy(const Tensor& src,
                             const Place& place,
                             bool blocking) {
  Tensor dst;
  copy(src, place, blocking, &dst);
  return std::make_shared<Tensor>(dst);
}

// std::shared_ptr<Tensor> copy(const Tensor* src, const Place& place, bool
// blocking) {
//   if (src == nullptr) return nullptr;
//   Tensor dst;
//   copy(*src, place, blocking, &dst);
//   return std::make_shared<Tensor>(dst);
// }

std::shared_ptr<std::vector<Tensor>> copy(const std::vector<Tensor>& src,
                                          const Place& place,
                                          bool blocking) {
  std::vector<Tensor> dst;
  copy(src, place, blocking, &dst);
  return std::make_shared<std::vector<Tensor>>(dst);
}

// std::shared_ptr<std::vector<Tensor*>> copy(const std::vector<Tensor*> src,
// const Place& place, bool blocking) {
//   std::vector<Tensor*> dst;
//   copy(src, place, blocking, dst);
//   return std::make_shared<std::vector<Tensor*>>(dst);
// }

std::shared_ptr<paddle::optional<Tensor>> copy(
    const paddle::optional<Tensor>& src, const Place& place, bool blocking) {
  Tensor dst_tmp;
  paddle::optional<Tensor> dst(dst_tmp);
  copy(src, place, blocking, &dst);
  return std::make_shared<paddle::optional<Tensor>>(dst);
}

std::shared_ptr<paddle::optional<std::vector<Tensor>>> copy(
    const optional<std::vector<Tensor>>& src,
    const Place& place,
    bool blocking) {
  std::vector<Tensor> dst_tmp;
  paddle::optional<std::vector<Tensor>> dst(dst_tmp);
  copy(src, place, blocking, &dst);
  return std::make_shared<paddle::optional<std::vector<Tensor>>>(dst);
}

double check_mse(const Tensor& a, const Tensor& b) {
  // return a.check_mse(b);
  if (a.is_dense_tensor()) {
    // return static_cast<phi::DenseTensor *>(impl_.get())->data<T>();
    // return std::to_string(
    //     static_cast<phi::DenseTensor*>(a.impl().get())
    //         ->check_mse(*(static_cast<phi::DenseTensor*>(b.impl().get()))));
    return static_cast<phi::DenseTensor*>(a.impl().get())
        ->check_mse(*(static_cast<phi::DenseTensor*>(b.impl().get())));
  } else if (a.is_selected_rows()) {
    // return static_cast<phi::SelectedRows *>(impl_.get())->value().data<T>();
    // return std::to_string(
    //     static_cast<phi::SelectedRows*>(a.impl().get())
    //         ->value()
    //         .check_mse(
    //             static_cast<phi::SelectedRows*>(b.impl().get())->value()));
    return static_cast<phi::SelectedRows*>(a.impl().get())
        ->value()
        .check_mse(static_cast<phi::SelectedRows*>(b.impl().get())->value());
  }
  // return "NOT_DENSETENSOER";
  return -1;
}

double check_mse(const paddle::optional<Tensor>& a,
                 const paddle::optional<Tensor>& b) {
  // return a ? check_mse(*a, *b) : "NOT_INIT";
  return a ? check_mse(*a, *b) : -2;
}

double check_mse(const std::vector<Tensor>& a, const std::vector<Tensor>& b) {
  double result = 0;
  for (size_t i = 0; i < a.size(); i++) {
    auto a_tensor = a[i];
    auto b_tensor = b[i];
    result += a_tensor.check_mse(b_tensor);
  }
  return a.size() > 0 ? result / a.size() : 0;
}

double check_mse(const optional<std::vector<Tensor>>& a,
                 const optional<std::vector<Tensor>>& b) {
  return a ? check_mse(*a, *b) : 0;
}

}  // namespace experimental
}  // namespace paddle
