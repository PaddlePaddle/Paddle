#pragma once

#include "paddle/phi/kernels/sparse/utils.h"

namespace phi {
namespace sparse {

DECLARE_SPARSE_UNARY_GRAD_KERNEL(Sqrt)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Sin)

}  // namespace sparse
}  // namespace phi
