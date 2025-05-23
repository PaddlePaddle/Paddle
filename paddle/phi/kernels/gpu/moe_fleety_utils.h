#pragma once

#include "paddle/extension.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/include/kernels.h"

namespace phi {

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      DenseTensor* out);

} // namespace phi

namespace fleety_utils {

namespace internal {

template <typename T>
struct TensorHasStrideImpl {
private:
    struct YesType {};   
    struct NoType {};

    template <typename U>
    static YesType Check(decltype(std::declval<U>().is_contiguous())) {
      return 0;
    }

    template <typename U>
    static NoType Check(...) {
      return 0;
    }

public:
    static constexpr bool kValue =
        std::is_same<YesType, decltype(Check<::phi::DenseTensorMeta>(false))>::value;
}; 


template <typename DenseT, typename PaddleT, bool _SupportStride> 
struct ContiguousTensorHelperImpl {
    static_assert(_SupportStride, "_SupportStride should be true");  

    static bool IsContiguousTensor(const DenseT &t) {
      return t.meta().is_contiguous();
    } 

    static typename std::enable_if<_SupportStride, void>::type TensorTrans2Contiguous(DenseT *t) {
      if (t != nullptr && t->initialized() && !t->meta().is_contiguous()) {
        auto place = t->place();
        auto is_gpu_place = place.GetType() == phi::AllocationType::GPU;
        PD_CHECK(is_gpu_place, "Only support GPU place");
        auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
        auto gpu_ctx = reinterpret_cast<const phi::GPUContext *>(dev_ctx); 
        auto dtype = t->dtype(); 

        PD_DISPATCH_FLOATING_AND_INTEGRAL_AND_COMPLEX_TYPES(dtype, "contiguous_kernel", ([&] {
          DenseT out;
          phi::ContiguousKernel<data_t, phi::GPUContext>(*gpu_ctx, *t, &out); 
          *t = out;
        }));
      }
    }

    static void TensorTrans2Contiguous(PaddleT *t) {
      if (t != nullptr) {
        if (!t->is_dense_tensor()) {
          PD_THROW("Trans2Contiguous only supports DenseTensor");
        }
        auto *dense_t = static_cast<DenseT *>(t->impl().get()); 
        TensorTrans2Contiguous(dense_t);
      }
    }
};


template <typename DenseT, typename PaddleT>
struct ContiguousTensorHelperImpl<DenseT, PaddleT, false> {
    static bool IsContiguousTensor(const DenseT &t) { return true; }
    static void TensorTrans2Contiguous(DenseT *t) {}
    static void TensorTrans2Contiguous(PaddleT *t) {}
};


} // namespace internal


inline constexpr bool SupportStride() {
    return internal::TensorHasStrideImpl<phi::DenseTensorMeta>::kValue;
} 

using ContiguousTensorHelper = internal::ContiguousTensorHelperImpl<phi::DenseTensor, paddle::Tensor, SupportStride()>;

inline bool IsContiguousTensor(const phi::DenseTensor &t) {
    return ContiguousTensorHelper::IsContiguousTensor(t);
} 

inline void TensorTrans2Contiguous(phi::DenseTensor *t) {
    return ContiguousTensorHelper::TensorTrans2Contiguous(t);
}

inline void TensorTrans2Contiguous(paddle::Tensor *t) {
    return ContiguousTensorHelper::TensorTrans2Contiguous(t);
}

} // namespace fleety_utils
