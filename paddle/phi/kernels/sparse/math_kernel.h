#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/sparse/utils.h"

namespace phi {
namespace sparse {

DECLARE_SPARSE_UNARY_KERNEL(Sqrt)
DECLARE_SPARSE_UNARY_KERNEL(Sin)

}  // namespace sparse
}  // namespace phi

