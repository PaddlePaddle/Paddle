#pragma once

#include <vector>

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

// used in new executor, for memory copy from host to device
template <typename Context>
void MemcpyH2DKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int dst_place_type,
                     DenseTensor* out);

// used in new executor, for memory copy from device to host
template <typename Context>
void MemcpyD2HKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int dst_place_type,
                     DenseTensor* out);

template <typename Context>
void MemcpyD2HMultiIOKernel(const Context& dev_ctx,
                            const std::vector<const DenseTensor*>& array,
                            int dst_place_type,
                            std::vector<DenseTensor*> out_array);

}  // namespace phi