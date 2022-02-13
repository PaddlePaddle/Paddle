

#pragma once

#include "paddle/pten/core/dense_tensor.h"
namespace pten {

template <typename T, typename Context>
void HistogramSelectKernel(const Context& dev_ctx,
                        const DenseTensor& input,
                        int64_t bins,
                        int min,
                        int max,
                        DenseTensor* out);

}  // namspace pten
