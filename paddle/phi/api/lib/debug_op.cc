/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/lib/debug_op.h"

// #include "paddle/utils/optional.h"
// #include "paddle/phi/api/lib/data_transform.h"
// #include "paddle/phi/api/lib/utils/allocator.h"
// #include "paddle/phi/backends/context_pool.h"
// #include "paddle/phi/api/lib/kernel_dispatch.h"
// #include "paddle/phi/api/lib/api_gen_utils.h"
// #include "paddle/phi/core/kernel_registry.h"
// #include "paddle/phi/core/tensor_utils.h"
// #include "paddle/phi/core/meta_tensor.h"
// #include "paddle/phi/infermeta/unary.h"
// #include "paddle/phi/kernels/cast_kernel.h"
// #include "paddle/phi/kernels/transfer_layout_kernel.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/backends/xpu/xpu_op_list.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace experimental {
static int64_t op_id = -1;

/* ------------------ for input ----------------------- */
// Copy for DenseTenso
std::shared_ptr<phi::DenseTensor> CopyDenseTensor(
    const std::shared_ptr<phi::DenseTensor>& in, Place dst_place) {
  if (in) {
    phi::DenseTensor out;
    // out.set_meta(in->meta());
    phi::MetaTensor meta_out(out);
    phi::UnchangedInferMeta(*in, &meta_out);
    auto& pool = paddle::experimental::DeviceContextPool::Instance();
    auto* dev_ctx = pool.GetMutable(dst_place);
    VLOG(6) << "start copy. ";
    if (in->initialized()) {
      phi::Copy(*dev_ctx, *in, dst_place, true, &out);
    } else {
      if (in->IsInitialized()) {
        out = phi::DenseTensor(std::make_shared<phi::Allocation>(
                                   nullptr, in->Holder()->size(), dst_place),
                               std::move(out.meta()));
        // out.ResetHolder(std::make_shared<phi::Allocation>(nullptr,
        // in->Holder()->size(), dst_place));
      }
    }
    VLOG(6) << "copy finished. ";

    VLOG(10) << "in->initialized(): " << in->initialized();
    VLOG(10) << "in->holder_?: " << in->IsInitialized();
    if (in->IsInitialized()) {
      VLOG(10) << "in->Holder()->ptr()?: " << bool(in->Holder()->ptr());
      VLOG(10) << "in->Holder()->size(): " << in->Holder()->size();
      VLOG(10) << "in->Holder()->place(): " << in->place();
    }
    VLOG(10) << "in->meta(): " << in->meta();

    VLOG(10) << "dst_place " << dst_place << std::endl;

    VLOG(10) << "out.initialized(): " << out.initialized();
    VLOG(10) << "out.holder_?: " << out.IsInitialized();
    if (out.IsInitialized()) {
      VLOG(10) << "out.Holder()->ptr()?: " << bool(out.Holder()->ptr());
      VLOG(10) << "out.Holder()->size(): " << out.Holder()->size();
      VLOG(10) << "out.Holder()->place(): " << out.place();
    }
    VLOG(10) << "out.meta(): " << out.meta();

    return std::make_shared<phi::DenseTensor>(std::move(out));
  }
  return nullptr;
}

paddle::optional<phi::DenseTensor> CopyDenseTensor(
    const paddle::optional<phi::DenseTensor>& in, Place dst_place) {
  if (in) {
    return {
        *CopyDenseTensor(std::make_shared<phi::DenseTensor>(*in), dst_place)};
  }
  return paddle::none;
}

// Copy for SelectedRows
std::shared_ptr<phi::SelectedRows> CopySelectedRows(
    const std::shared_ptr<phi::SelectedRows>& in, Place dst_place) {
  if (in) {
    phi::SelectedRows out;
    // out.set_meta(in->meta());
    phi::MetaTensor meta_out(out);
    phi::UnchangedInferMeta(*in, &meta_out);
    auto& pool = paddle::experimental::DeviceContextPool::Instance();
    auto* dev_ctx = pool.GetMutable(dst_place);
    VLOG(6) << "start copy. ";
    if (in->initialized()) {
      phi::Copy(*dev_ctx, *in, dst_place, true, &out);
    }
    // else {
    //     if (in->IsInitialized()) {
    //       // out =
    //       phi::DenseTensor(std::make_shared<phi::Allocation>(nullptr,
    //       in->Holder()->size(), dst_place),
    //       //       std::move(out.meta()));
    //       out.ResetHolder(std::make_shared<phi::Allocation>(nullptr,
    //       in->Holder()->size(), dst_place));
    //     };
    // }
    VLOG(6) << "copy finished. ";

    VLOG(10) << "in->initialized(): " << in->initialized();
    VLOG(10) << "in->holder_?: " << in->value().IsInitialized();
    if (in->value().IsInitialized()) {
      VLOG(10) << "in->Holder()->ptr()?: " << bool(in->value().Holder()->ptr());
      VLOG(10) << "in->Holder()->size(): " << in->value().Holder()->size();
      VLOG(10) << "in->Holder()->place(): " << in->place();
    }
    VLOG(10) << "in->meta(): " << in->value().meta();

    VLOG(10) << "dst_place " << dst_place << std::endl;

    VLOG(10) << "out.initialized(): " << out.initialized();
    VLOG(10) << "out.holder_?: " << out.value().IsInitialized();
    if (out.value().IsInitialized()) {
      VLOG(10) << "out.Holder()->ptr()?: " << bool(out.value().Holder()->ptr());
      VLOG(10) << "out.Holder()->size(): " << out.value().Holder()->size();
      VLOG(10) << "out.Holder()->place(): " << out.place();
    }
    VLOG(10) << "out.meta(): " << out.value().meta();

    return std::make_shared<phi::SelectedRows>(std::move(out));
  }
  return nullptr;
}

paddle::optional<phi::SelectedRows> CopySelectedRows(
    const paddle::optional<phi::SelectedRows>& in, Place dst_place) {
  if (in) {
    return {
        *CopySelectedRows(std::make_shared<phi::SelectedRows>(*in), dst_place)};
  }
  return paddle::none;
}

// std::unique_ptr<std::vector<phi::DenseTensor>> CopyVector(
//     const std::unique_ptr<std::vector<phi::DenseTensor>>& ins,
//     Place dst_place) {
//   auto outs = std::make_unique<std::vector<phi::DenseTensor>>();
//   outs->reserve(ins->size());
//   for (const auto& in : *ins) {
//     auto out = Copy(in, dst_place);
//     outs->emplace_back(std::move(*out));
//   }
//   return outs;
// }

std::unique_ptr<std::vector<phi::DenseTensor>> CopyVector(
    const std::vector<const phi::DenseTensor*>& ins, Place dst_place) {
  auto outs = std::make_unique<std::vector<phi::DenseTensor>>();
  outs->reserve(ins.size());
  for (const auto& in : ins) {
    if (in == nullptr) {
      VLOG(10) << "in == nullptr ? : " << (in == nullptr);
      auto out = std::make_shared<phi::DenseTensor>();
      // out->set_empty();
      outs->emplace_back(std::move(*out));
    } else {
      auto out = CopyDenseTensor(*in, dst_place);
      outs->emplace_back(std::move(*out));
    }
  }
  return outs;
}

// paddle::optional<std::vector<phi::DenseTensor>> CopyOptionalVector(
//     const paddle::optional<std::vector<phi::DenseTensor>>& ins,
//     Place dst_place) {
//   if (ins) {
//     return {*Copy(*ins, dst_place)};
//   }
//   return paddle::none;
// }

paddle::optional<std::vector<phi::DenseTensor>> CopyOptionalVector(
    const paddle::optional<std::vector<const phi::DenseTensor*>>& ins,
    Place dst_place) {
  if (ins) {
    return {*CopyVector(*ins, dst_place)};
  }
  return paddle::none;
}

std::vector<const phi::DenseTensor*> DenseTensorToConstDenseTensorPtr(
    const std::vector<phi::DenseTensor>& tensors,
    const std::vector<const phi::DenseTensor*>& ins) {
  std::vector<const phi::DenseTensor*> pt_tensors(tensors.size(), nullptr);

  for (size_t i = 0; i < tensors.size(); ++i) {
    if (ins[i]) {
      pt_tensors[i] = &tensors[i];
    }
  }

  return pt_tensors;
}

paddle::optional<std::vector<const phi::DenseTensor*>>
DenseTensorToConstDenseTensorPtr(
    const paddle::optional<std::vector<phi::DenseTensor>>& tensors) {
  paddle::optional<std::vector<const phi::DenseTensor*>> pt_tensors;

  if (tensors) {
    pt_tensors =
        paddle::optional<std::vector<const phi::DenseTensor*>>(tensors->size());
    for (size_t i = 0; i < tensors->size(); ++i) {
      pt_tensors->at(i) = &tensors->at(i);
    }
  }

  return pt_tensors;
}

/* ------------------ for output ----------------------- */
std::shared_ptr<phi::DenseTensor> CopyDenseTensor(const phi::DenseTensor* in,
                                                  Place dst_place) {
  if (in) {
    return CopyDenseTensor(std::make_shared<phi::DenseTensor>(*in), dst_place);
  }
  return nullptr;
}

std::shared_ptr<phi::SelectedRows> CopySelectedRows(const phi::SelectedRows* in,
                                                    Place dst_place) {
  if (in) {
    return CopySelectedRows(std::make_shared<phi::SelectedRows>(*in),
                            dst_place);
  }
  return nullptr;
}

std::unique_ptr<std::vector<phi::DenseTensor>> CopyVector(
    const std::vector<phi::DenseTensor*>& ins, Place dst_place) {
  auto outs = std::make_unique<std::vector<phi::DenseTensor>>();
  outs->reserve(ins.size());
  for (const auto& in : ins) {
    if (in == nullptr) {
      VLOG(10) << "in == nullptr ? : " << (in == nullptr);
      auto out = std::make_shared<phi::DenseTensor>();
      // out->set_empty();
      outs->emplace_back(std::move(*out));
    } else {
      auto out = CopyDenseTensor(*in, dst_place);
      outs->emplace_back(std::move(*out));
    }
  }
  return outs;
}

// paddle::optional<std::vector<phi::DenseTensor>> CopyOptionalVector(
//     const paddle::optional<std::vector<phi::DenseTensor*>>& ins,
//     Place dst_place) {
//   if (ins) {
//     return {*CopyVector(*ins, dst_place)};
//   }
//   return paddle::none;
// }

std::vector<phi::DenseTensor*> DenseTensorToDenseTensorPtr(
    std::vector<phi::DenseTensor>* tensors,
    const std::vector<phi::DenseTensor*>& ins) {
  std::vector<phi::DenseTensor*> pt_tensors(tensors->size(), nullptr);

  for (size_t i = 0; i < tensors->size(); ++i) {
    if (ins[i]) {
      pt_tensors[i] = &tensors->at(i);
    }
    // pt_tensors[i] = static_cast<phi::DenseTensor*>(&tensors[i]);
  }

  return pt_tensors;
}

std::vector<phi::DenseTensor*> DenseTensorToDenseTensorPtr(
    std::vector<phi::DenseTensor>* tensors,
    const std::vector<const phi::DenseTensor*>& ins) {
  std::vector<phi::DenseTensor*> pt_tensors(tensors->size(), nullptr);

  for (size_t i = 0; i < tensors->size(); ++i) {
    if (ins[i]) {
      pt_tensors[i] = &tensors->at(i);
    }
    // pt_tensors[i] = static_cast<phi::DenseTensor*>(&tensors[i]);
  }

  return pt_tensors;
}

std::vector<phi::DenseTensor*> DenseTensorToDenseTensorPtr(
    const paddle::optional<std::vector<phi::DenseTensor>>& tensors) {
  std::vector<phi::DenseTensor*> pt_tensors;

  if (tensors) {
    pt_tensors = std::vector<phi::DenseTensor*>(tensors->size(), nullptr);
    for (size_t i = 0; i < tensors->size(); ++i) {
      pt_tensors[i] = const_cast<phi::DenseTensor*>(&tensors->at(i));
    }
  }

  return pt_tensors;
}

// phi::DenseTensor* DenseTensorToDenseTensorPtr(phi::DenseTensor* out) {}

// std::vector<phi::DenseTensor*>
// DenseTensorToDenseTensorPtr(std::vector<phi::DenseTensor>& tensors) {}

// phi::DenseTensor* SetSelectedRowsToDenseTensorPtr(phi::DenseTensor* out) {}

/* ------------------ for device 2 ----------------------- */
phi::Place& GetDebugDev2Type() {
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
    VLOG(3) << "XPU Paddle Debug Dev2 Type: " << device;
  }
  return dev2;
}

/* ------------------ for check acc ----------------------- */

// Mean Squared Error
struct MSEFunctor {
  template <typename T>
  double operator()(T* val_dev1, T* val_dev2, size_t size) {
    double result = 0.0;
    for (size_t i = 0; i < size; ++i) {
      // result += (val[i] / a.numel());
      result += pow(
          static_cast<double>(val_dev1[i]) - static_cast<double>(val_dev2[i]),
          2);
    }
    return result / size;
  }
};

// Euclidean Distance
struct MDFunctor {
  template <typename T>
  double operator()(T* val_dev1, T* val_dev2, size_t size) {
    double result = 0.0;
    for (size_t i = 0; i < size; ++i) {
      // result += (val[i] / a.numel());
      result += pow(
          static_cast<double>(val_dev1[i]) - static_cast<double>(val_dev2[i]),
          2);
    }
    return std::sqrt(result);
  }
};

// Mean Relative Error
struct MREFunctor {
  template <typename T>
  double operator()(T* val_dev1, T* val_dev2, size_t size) {
    double sum_relative_error = 0;
    size_t non_zero_count = 0;
    for (size_t i = 0; i < size; ++i) {
      if (val_dev1[i] != static_cast<T>(0)) {
        double relative_error = std::abs((static_cast<double>(val_dev1[i]) -
                                          static_cast<double>(val_dev2[i])) /
                                         static_cast<double>(val_dev1[i]));
        sum_relative_error += relative_error;
        non_zero_count++;
      }
    }
    return non_zero_count > 0 ? sum_relative_error / non_zero_count : 0;
  }
};

// Maximum Absolute Error
struct MAEFunctor {
  template <typename T>
  double operator()(T* val_dev1, T* val_dev2, size_t size) {
    double max_error = 0;
    for (size_t i = 0; i < size; ++i) {
      double error = std::abs(static_cast<double>(val_dev1[i]) -
                              static_cast<double>(val_dev2[i]));
      if (error > max_error) {
        max_error = error;
      }
    }
    return max_error;
  }
};

template <typename T, typename... Functor>
std::vector<double> CheckAccImpl(const phi::DenseTensor& a,
                                 const phi::DenseTensor& b) {
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  VLOG(10) << "tensor A meta: " << a.meta();
  VLOG(10) << "tensor B meta: " << b.meta();
  T* val_dev1 = const_cast<T*>(a.data<T>());
  T* val_dev2 = const_cast<T*>(b.data<T>());
  VLOG(10) << "dev1_place: " << a.place() << ", dev1_addr: " << a.data()
           << ", dev1_numel: " << a.numel();

  VLOG(10) << "dev2_place: " << b.place() << ", dev2_addr: " << b.data()
           << ", dev2_numel: " << b.numel();
  if (paddle::platform::is_xpu_place(a.place())) {
    auto& dev_ctx = *pool.Get(a.place());
    val_dev1 = new T[a.numel()];
    dev_ctx.Wait();
    xpu_memcpy(val_dev1, a.data(), a.numel() * sizeof(T), XPU_DEVICE_TO_HOST);
  }

  if (paddle::platform::is_xpu_place(b.place())) {
    auto& dev_ctx = *pool.Get(b.place());
    val_dev2 = new T[b.numel()];
    dev_ctx.Wait();
    xpu_memcpy(val_dev2, b.data(), b.numel() * sizeof(T), XPU_DEVICE_TO_HOST);
  }
  // MSEFunctor mse_func;
  // results.emplace_back(mse_func(val_dev1, val_dev2));
  // for (int i = 0; i < a.numel(); i++) {
  //   // result += (val[i] / a.numel());
  //   result +=
  //       pow(static_cast<double>(val_dev1[i]) -
  //       static_cast<double>(val_dev2[i]), 2);
  // }
  std::vector<double> result{Functor()(val_dev1, val_dev2, a.numel())...};
  if (paddle::platform::is_xpu_place(a.place())) {
    delete[] val_dev1;
  }

  if (paddle::platform::is_xpu_place(b.place())) {
    delete[] val_dev2;
  }
  VLOG(10) << "Dense tensor check mse end.";
  // VLOG(10) << "result = " << result;
  return result;
  // return result / a.numel();
}

template <typename... Functor>
std::vector<double> CheckAcc(const phi::DenseTensor& a,
                             const phi::DenseTensor& b) {
  if (!a.initialized()) {
    return {-4};
  }
  PADDLE_ENFORCE_EQ(a.dtype(),
                    b.dtype(),
                    phi::errors::InvalidArgument(
                        "The type of tensor A data (%s) does not match the "
                        "type of tensor B data (%s).",
                        a.dtype(),
                        b.dtype()));
  PADDLE_ENFORCE_EQ(a.numel(),
                    b.numel(),
                    phi::errors::InvalidArgument(
                        "The numel of tensor A data (%s) does not match the "
                        "numel of tensor B data (%s).",
                        a.numel(),
                        b.numel()));

  switch (a.dtype()) {
    case phi::CppTypeToDataType<bool>::Type():
      return CheckAccImpl<bool, Functor...>(a, b);
    case phi::CppTypeToDataType<int8_t>::Type():
      return CheckAccImpl<int8_t, Functor...>(a, b);
    case phi::CppTypeToDataType<uint8_t>::Type():
      return CheckAccImpl<uint8_t, Functor...>(a, b);
    case phi::CppTypeToDataType<int16_t>::Type():
      return CheckAccImpl<int16_t, Functor...>(a, b);
    case phi::CppTypeToDataType<uint16_t>::Type():
      return CheckAccImpl<uint16_t, Functor...>(a, b);
    case phi::CppTypeToDataType<int32_t>::Type():
      return CheckAccImpl<int32_t, Functor...>(a, b);
    case phi::CppTypeToDataType<uint32_t>::Type():
      return CheckAccImpl<uint32_t, Functor...>(a, b);
    case phi::CppTypeToDataType<int64_t>::Type():
      return CheckAccImpl<int64_t, Functor...>(a, b);
    case phi::CppTypeToDataType<uint64_t>::Type():
      return CheckAccImpl<uint64_t, Functor...>(a, b);
    case phi::CppTypeToDataType<phi::float16>::Type():
      return CheckAccImpl<phi::float16, Functor...>(a, b);
    case phi::CppTypeToDataType<float>::Type():
      return CheckAccImpl<float, Functor...>(a, b);
    case phi::CppTypeToDataType<double>::Type():
      return CheckAccImpl<double, Functor...>(a, b);
    default:
      VLOG(10) << "Not support data type: " << a.dtype();
      return {-3};
  }
}

/* ------------------ for parsing environment variables -----------------------
 */
int64_t OpId() { return op_id; }
// int64_t OpIdAdd() { return op_id++; }
int64_t OpIdAdd() {
  static std::mutex s_mtx;
  std::lock_guard<std::mutex> guard(s_mtx);
  op_id++;
  return op_id;
}

bool ContinueOrNot(const std::string& op_name) {
  auto fluid_name = phi::TransToFluidOpName(op_name);
  bool continue_or_not =
      !phi::backends::xpu::is_in_xpu_debug_black_list(fluid_name) &&
      !phi::backends::xpu::is_in_xpu_debug_black_id_list(
          std::to_string(OpId()));
  continue_or_not =
      continue_or_not &&
      (phi::backends::xpu::is_in_xpu_debug_white_list(fluid_name) ||
       std::getenv("XPU_PADDLE_DEBUG_WHITE_LIST") == nullptr);
  continue_or_not = continue_or_not &&
                    (phi::backends::xpu::is_in_xpu_debug_white_id_list(
                         std::to_string(OpId())) ||
                     std::getenv("XPU_PADDLE_DEBUG_WHITE_ID_LIST") == nullptr);
  return continue_or_not;
}

// bool ContinueRunDev2OrNot(const std::string& op_name) {
//   auto fluid_name = phi::TransToFluidOpName(op_name);
//   bool continue_or_not =
//       !phi::backends::xpu::is_in_xpu_debug_run_dev2_black_list(fluid_name);
//   return continue_or_not;
// }

bool DebugOrNot() {
  bool continue_or_not = (std::getenv("XPU_PADDLE_DEBUG_GLOBAL") != nullptr ||
                          std::getenv("XPU_PADDLE_DEBUG_OP") != nullptr);
  return continue_or_not;
}

static void tokenize(const std::string& ops,
                     char delim,
                     std::vector<double>* op_set) {
  std::string::size_type beg = 0;
  // double thres = -10;
  size_t count = 0;
  for (uint64_t end = 0; (end = ops.find(delim, end)) != std::string::npos;
       ++end) {
    op_set->at(count) = std::stod(ops.substr(beg, end - beg));
    beg = end + 1;
    count++;
  }

  op_set->at(count) = std::stod(ops.substr(beg));
}

std::vector<double> GetDebugThres(size_t size) {
  // static double thres = -10;
  static bool inited = false;
  static std::vector<double> xpu_debug_thres_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      xpu_debug_thres_list = std::vector<double>(size, -10);
      if (std::getenv("XPU_PADDLE_DEBUG_THRES") != nullptr) {
        std::string ops(std::getenv("XPU_PADDLE_DEBUG_THRES"));
        tokenize(ops, ',', &xpu_debug_thres_list);
        // std::istringstream isthres_str(thres_str);
        // isthres_str >> thres;
      }
      inited = true;
      VLOG(3) << "XPU Paddle Debug Check Threshold: ";
      for (auto iter = xpu_debug_thres_list.begin();
           iter != xpu_debug_thres_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  return xpu_debug_thres_list;
}

/* ------------------ for log acc ----------------------- */
std::string XPUDebugStartString(const std::string& op_name,
                                const Backend& dev_place,
                                const DataType& kernel_data_type) {
  if (ContinueOrNot(op_name)) {
    std::stringstream print_buffer;
    print_buffer << "op_name_debug " << phi::TransToFluidOpName(op_name) << " "
                 << OpId() << " " << kernel_data_type << " ";
    //  << dev_place
    //  << " ";
    //  << kernel_data_type << " in: ";
    return print_buffer.str();
  } else {
    return "";
  }
}

static std::pair<std::string, bool> XPUDebugStringImpl(
    const std::string& tensor_name,
    const phi::DenseTensor& a,
    const phi::DenseTensor& b) {
  std::stringstream print_buffer;
  bool thes_open = false;
  // print_buffer << tensor_name;
  // VLOG(10) << print_buffer.str();
  if (a.initialized()) {
    VLOG(11) << tensor_name << " " << a.dtype() << " " << a.place();
    VLOG(11) << "dev2_" << tensor_name << " " << b.dtype() << " " << b.place();
    print_buffer << "-dev1:[" << a.dtype() << "," << a.place().DebugString()
                 << "]"
                 << "-dev2:[" << b.dtype() << "," << b.place().DebugString()
                 << "]"
                 << "-acc:[";
    auto accs = CheckAcc<MSEFunctor, MAEFunctor, MREFunctor>(a, b);
    auto thres = GetDebugThres(accs.size());
    for (size_t i = 0; i < accs.size(); i++) {
      if (accs[i] > thres[i]) {
        thes_open = true;
      }
      if (i < accs.size() - 1) {
        print_buffer << accs[i] << ",";
      } else {
        print_buffer << accs[i];
      }
    }
    print_buffer << "]";
  } else {
    VLOG(11) << tensor_name << " DenseTensor not initial";
    print_buffer << "-NOT_INITED";
  }
  // VLOG(10) << print_buffer.str();
  return std::make_pair(print_buffer.str(), thes_open);
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const phi::DenseTensor& a,
                           const phi::DenseTensor& b) {
  auto tmp = XPUDebugStringImpl(tensor_name, a, b);
  // std::string tmp = XPUDebugStringImpl(tensor_name, a, b);
  VLOG(10) << tensor_name + " " + tmp.first;
  if (tmp.second) {
    return tensor_name + tmp.first + " ";
  } else {
    return "";
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const paddle::optional<phi::DenseTensor>& a,
                           const paddle::optional<phi::DenseTensor>& b) {
  if (a) {
    auto tmp = XPUDebugStringImpl(tensor_name, *a, *b);
    VLOG(10) << tensor_name + tmp.first;
    if (tmp.second) {
      return tensor_name + tmp.first + " ";
    } else {
      return "";
    }
  } else {
    bool flag = false;
    VLOG(10) << tensor_name << " Optional DenseTensor is none";
    for (auto& i : GetDebugThres(3)) {
      if (i < -2) {
        flag = true;
      }
    }
    if (flag) {
      return tensor_name + "-NOT_INITED_OPTIONAL ";
    } else {
      return "";
    }
  }
  // return a ? XPUDebugString(op_name, tensor_name, *a, *b) : tensor_name +
  // "-NOT_INITED_OPTIONAL ";
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const phi::SelectedRows& a,
                           const phi::SelectedRows& b) {
  auto tmp = XPUDebugStringImpl(tensor_name, a.value(), b.value());
  // std::string tmp = XPUDebugStringImpl(tensor_name, a, b);
  VLOG(10) << tensor_name + tmp.first;
  if (tmp.second) {
    return tensor_name + tmp.first + " ";
  } else {
    return "";
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const paddle::optional<phi::SelectedRows>& a,
                           const paddle::optional<phi::SelectedRows>& b) {
  if (a) {
    return XPUDebugString(op_name, tensor_name, *a, *b);
  } else {
    VLOG(10) << tensor_name << " Optional SelectedRows is none";
    bool flag = false;
    for (auto& i : GetDebugThres(3)) {
      if (i < -2) {
        flag = true;
      }
    }
    if (flag) {
      return tensor_name + "-NOT_INITED_OPTIONAL ";
    } else {
      return "";
    }
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const std::vector<const phi::DenseTensor*>& a,
                           const std::vector<const phi::DenseTensor*>& b) {
  VLOG(10) << "a.size() = " << a.size() << ", b.size() = " << b.size();
  PADDLE_ENFORCE_EQ(a.size(),
                    b.size(),
                    phi::errors::InvalidArgument(
                        "The size of vector A (%s) does not match the "
                        "size of tensor B (%s).",
                        a.size(),
                        b.size()));
  std::string str = "-vector{";
  bool thres_open = false;
  bool flag = false;
  for (auto& i : GetDebugThres(3)) {
    if (i < -2) {
      flag = true;
    }
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i]) {
      auto tmp = XPUDebugStringImpl(tensor_name, *a[i], *b[i]);
      if (tmp.second) {
        thres_open = true;
        str += (std::to_string(i) + tmp.first + ",");
      }
    } else {
      if (flag) {
        str += (std::to_string(i) + "-NOT_INITED,");
      }
    }
  }
  str = str.substr(0, str.length() - 1) + "} ";
  VLOG(10) << tensor_name + str;
  if (!thres_open && !flag) {
    return "";
  } else {
    return tensor_name + str;
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const std::vector<const phi::TensorBase*>& a,
                           const std::vector<const phi::TensorBase*>& b) {
  VLOG(10) << "a.size() = " << a.size() << ", b.size() = " << b.size();
  PADDLE_ENFORCE_EQ(a.size(),
                    b.size(),
                    phi::errors::InvalidArgument(
                        "The size of vector A (%s) does not match the "
                        "size of tensor B (%s).",
                        a.size(),
                        b.size()));
  std::string str = "-vector{";
  bool thres_open = false;
  bool flag = false;
  for (auto& i : GetDebugThres(3)) {
    if (i < -2) {
      flag = true;
    }
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i]) {
      std::pair<std::string, bool> tmp = {"", false};
      if (phi::DenseTensor::classof(a[i])) {
        tmp = XPUDebugStringImpl(tensor_name,
                                 *static_cast<const phi::DenseTensor*>(a[i]),
                                 *static_cast<const phi::DenseTensor*>(b[i]));
      } else if (phi::SelectedRows::classof(a[i])) {
        tmp = XPUDebugStringImpl(
            tensor_name,
            static_cast<const phi::SelectedRows*>(a[i])->value(),
            static_cast<const phi::SelectedRows*>(b[i])->value());
      }
      if (tmp.second) {
        thres_open = true;
        str += (std::to_string(i) + tmp.first + ",");
      }
    } else {
      if (flag) {
        str += (std::to_string(i) + "-NOT_INITED,");
      }
    }
  }
  str = str.substr(0, str.length() - 1) + "} ";
  VLOG(10) << tensor_name + str;
  if (!thres_open && !flag) {
    return "";
  } else {
    return tensor_name + str;
  }
}

std::string XPUDebugString(
    const std::string& op_name,
    const std::string& tensor_name,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& a,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& b) {
  if (a) {
    return XPUDebugString(op_name, tensor_name, *a, *b);
  } else {
    VLOG(11) << tensor_name << " Optional vector is none";
    bool flag = false;
    for (auto& i : GetDebugThres(3)) {
      if (i < -2) {
        flag = true;
      }
    }
    if (flag) {
      return tensor_name + "-NOT_INITED_OPTIONAL ";
    } else {
      return "";
    }
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const phi::DenseTensor* a,
                           const phi::DenseTensor* b) {
  if (a) {
    auto tmp = XPUDebugStringImpl(tensor_name, *a, *b);
    VLOG(10) << tensor_name + tmp.first;
    if (tmp.second) {
      return tensor_name + tmp.first + " ";
    } else {
      return "";
    }
  } else {
    bool flag = false;
    VLOG(11) << tensor_name << " DenseTensor ptr is nullptr";
    for (auto& i : GetDebugThres(3)) {
      if (i < 0) {
        flag = true;
      }
    }
    if (flag) {
      return tensor_name + "-NOT_INITED_PTR ";
    } else {
      return "";
    }
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const phi::SelectedRows* a,
                           const phi::SelectedRows* b) {
  if (a) {
    auto tmp = XPUDebugStringImpl(tensor_name, a->value(), b->value());
    VLOG(10) << tensor_name + tmp.first;
    if (tmp.second) {
      return tensor_name + tmp.first + " ";
    } else {
      return "";
    }
  } else {
    bool flag = false;
    VLOG(11) << tensor_name << " SelectedRows ptr is nullptr";
    for (auto& i : GetDebugThres(3)) {
      if (i < 0) {
        flag = true;
      }
    }
    if (flag) {
      return tensor_name + "-NOT_INITED_PTR ";
    } else {
      return "";
    }
  }
}

std::string XPUDebugString(const std::string& op_name,
                           const std::string& tensor_name,
                           const std::vector<phi::DenseTensor*>& a,
                           const std::vector<phi::DenseTensor*>& b) {
  VLOG(10) << "a.size() = " << a.size() << ", b.size() = " << b.size();
  PADDLE_ENFORCE_EQ(a.size(),
                    b.size(),
                    phi::errors::InvalidArgument(
                        "The size of vector A (%s) does not match the "
                        "size of tensor B (%s).",
                        a.size(),
                        b.size()));
  std::string str = "-vector{";
  bool thres_open = false;
  bool flag = false;
  for (auto& i : GetDebugThres(3)) {
    if (i < -2) {
      flag = true;
    }
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i]) {
      auto tmp = XPUDebugStringImpl(tensor_name, *a[i], *b[i]);
      if (tmp.second) {
        thres_open = true;
        str += (std::to_string(i) + tmp.first + ",");
      }
    } else {
      if (flag) {
        str += (std::to_string(i) + "-NOT_INITED,");
      }
    }
  }
  str = str.substr(0, str.length() - 1) + "} ";
  VLOG(10) << tensor_name + str;
  if (!thres_open && !flag) {
    return "";
  } else {
    return tensor_name + str;
  }
}
}  // namespace experimental
}  // namespace paddle
