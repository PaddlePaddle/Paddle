
#pragma once

#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
struct LsqplusFakequant {
    inline HOSTDEVICE void operator()(size_t i) {
        T x_val = x_[i];
        T alpha = alpha_[0]; // alpha need to be a scalar
        T beta = beta_[0]; // beta need to be a scalar
        out_[i] = round((x_val-beta)/alpha);
        out_[i] = out_[i] > Qp_ ? Qp_ : out_[i];
        out_[i] = out_[i] < Qn_ ? Qn_ : out_[i];
        out_[i] = out_[i]*alpha+beta;
    };

    const T* x_;
    const T* alpha_;
    const T* beta_;
    const int Qn_;
    const int Qp_;
    T* out_;
};


template <typename T, typename Context>
void LsqplusKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& alpha,
    const DenseTensor& beta,
    const DenseTensor& g,
    int Qn,
    int Qp,
    DenseTensor* out);

}