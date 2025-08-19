#pragma once
#include <ATen/core/TensorBody.h>

namespace at::detail {
using namespace c10;

at::Tensor empty_cuda(
    IntArrayRef size,
    ScalarType dtype,
    std::optional<Device> device_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

// at::Tensor empty_cuda(
//     IntArrayRef size,
//     std::optional<ScalarType> dtype_opt,
//     std::optional<Layout> layout_opt,
//     std::optional<Device> device_opt,
//     std::optional<bool> pin_memory_opt,
//     std::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor empty_cuda(
    IntArrayRef size,
    const TensorOptions &options);

}  // namespace at::detail
