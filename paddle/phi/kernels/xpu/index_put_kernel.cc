#include "paddle/phi/kernels/index_put_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/stack_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"

namespace phi {
template <typename Context>
void XPUDealWithIndices(const Context& dev_ctx,
                        const std::vector<const DenseTensor*>& int_indices_v,
                        DDim bd_dim,
                        DenseTensor* out) {
  std::vector<DenseTensor> tmp_indices_v;
  for (size_t i = 0; i < int_indices_v.size(); ++i) {
    DenseTensor casted_index;
    if (int_indices_v[i]->dtype() == DataType::INT32) {
      casted_index = phi::Cast<int, Context>(dev_ctx, *int_indices_v[i], DataType::INT64);
    } else {
      casted_index = *int_indices_v[i];
    }

    DenseTensor expanded_index(DataType::INT64);
    if (casted_index.dims() == bd_dim) {
      expanded_index = casted_index;
    } else {
      expanded_index.Resize(bd_dim);
      ExpandKernel<int64_t, Context>(dev_ctx, casted_index, IntArray(vectorize<int64_t>(bd_dim)), &expanded_index);
    }

    tmp_indices_v.emplace_back(expanded_index);
  }

  auto bd_dim_vec = vectorize<int64_t>(bd_dim);
  std::vector<int64_t> stacked_dim_vec(bd_dim.size() + 1);
  std::copy(bd_dim_vec.begin(), bd_dim_vec.end(), stacked_dim_vec.begin());
  stacked_dim_vec.back() = int_indices_v.size();
  out->Resize(make_ddim(stacked_dim_vec));

  std::vector<const DenseTensor*> tmp_indices_ptr(tmp_indices_v.size(), nullptr);
  for (size_t i = 0; i < tmp_indices_ptr.size(); ++i) {
    tmp_indices_ptr[i] = &tmp_indices_v[i];
  }

  StackKernel<int64_t, Context>(dev_ctx, tmp_indices_ptr, -1, out);
}

template <typename T, typename Context>
void IndexPutKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices,
                    const DenseTensor& value,
                    bool accumulate,
                    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
    x.dtype(),
    value.dtype(),
    phi::errors::InvalidArgument(
        "The data type of tensor value must be same to the data type "
        "of tensor x."));
  PADDLE_ENFORCE_EQ(indices.empty(),
                    false,
                    phi::errors::InvalidArgument("Indices cannot be empty."));
  const int64_t total_dims = x.dims().size();
  PADDLE_ENFORCE_LE(total_dims,
                    6,
                    errors::InvalidArgument("Dims of input tensor should be less than 7."));
  
  std::vector<DenseTensor> tmp_args;
  std::vector<const DenseTensor*> int_indices_v = funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &tmp_args);
  if (int_indices_v.empty()) {
    if (!out->initialized()) {
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }

  using XPUT = typename XPUTypeTrait<T>::Type;
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto bd_dims = funcs::BroadCastTensorsDims(int_indices_v);
  DenseTensor res_indices(DataType::INT64);
  XPUDealWithIndices<Context>(dev_ctx, int_indices_v, bd_dims, &res_indices);
  auto index_shape = vectorize<int64_t>(res_indices.dims());
  auto x_shape = vectorize<int64_t>(x.dims());
  const T* value_data = value.data<T>();

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  if (value.numel() == 1) {
    int64_t value_rank = bd_dims.size() + (x_shape.size() - int_indices_v.size());
    std::vector<int64_t> value_shape(value_rank);
    std::copy(index_shape.begin(), index_shape.end(), value_shape.begin());
    std::copy(x_shape.begin() + int_indices_v.size(), x_shape.end(), value_shape.begin() + index_shape.size());
    auto value_data_bd = RAII_GUARD.alloc_l3_or_gm<T>(product(make_ddim(value_shape)));
    int r = xpu::broadcast<XPUT>(dev_ctx.x_context(),
                                 reinterpret_cast<const XPUT*>(value_data),
                                 reinterpret_cast<XPUT*>(value_data_bd),
                                 {1},
                                 value_shape);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
    value_data = value_data_bd;
  }
  
  int r = xpu::index_put<XPUT, int64_t>(dev_ctx.x_context(),
                                        reinterpret_cast<const XPUT*>(x.data<T>()),
                                        reinterpret_cast<const XPUT*>(value_data),
                                        res_indices.data<int64_t>(),
                                        reinterpret_cast<XPUT*>(out_data),
                                        x_shape,
                                        index_shape,
                                        accumulate);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_put");
}
}  // namespace phi

PD_REGISTER_KERNEL(index_put,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexPutKernel,
                   float,
                   int,
                   int64_t) {}

