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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/sparse/concat_grad_kernel.h"

namespace phi {
namespace sparse {
static inline int GetDimensionInSlice(const int dim,
                                      const int slice_index,
                                      std::vector<int> slice_index_chowks) {
  if (slice_index == 0) return dim;
  return dim % slice_index_chowks[slice_index - 1];
}

template <typename T>
void split(const SparseCooTensor& input_tensor,
           const int split_dim,
           const std::vector<const phi::SparseCooTensor*>& ref_inputs,
           std::vector<SparseCooTensor*> result) {
  std::vector<DenseTensor> output_indices;
  std::vector<DenseTensor> output_values;
  std::vector<DDim> output_shapes;

  const size_t num_split = ref_inputs.size();
  output_indices.reserve(num_split);
  output_values.reserve(num_split);
  output_shapes.reserve(num_split);

  std::vector<int64_t*> output_indices_t;
  std::vector<T*> output_values_t;

  output_indices_t.reserve(num_split);
  output_values_t.reserve(num_split);
  T* input_values_t = input_tensor.values().data<T>();
  // auto input_values_t = input_tensor.values().vec<T>();
  int64_t* input_indices_t = input_tensor.indices().data<int64_t>();
  // auto input_indices_t = input_tensor.indices().matrix<int64_t>();

  std::vector<int> num_values(num_split, 0);

  const int split_dim_size = input_tensor.dims()[split_dim];
  // 有这个函数的地方进行修改
  // 实际不会用到,最后删除 理论上每一个划分出来的稀疏矩阵的在split域上的(长度)
  const int split_size = split_dim_size / num_split;

  if (!(num_split > 0 && num_split <= split_dim_size)) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "num_split must be in the interval (0, ", split_dim_size, "]"));
  }
  if (!(split_dim >= 0 && split_dim < num_dim)) {
    PADDLE_THROW("num_dim must be in the interval [0, ", num_dim, ")");
  }

  int64_t total_nnz = 0;
  int64_t total_values_nnz = 0;
  std::vector<int> slice_index_chowks;
  for (size_t i = 0; i < num_split; ++i) {
    int64_t now_nnz = ref_inputs[i]->indices().dims()[1];
    num_values[i] = now_nnz;
    total_nnz += now_nnz;
    total_values_nnz += ref_inputs[i]->values().dims()[0];
    slice_index_chowks.push_back(total_nnz);
    output_indices.emplace_back(
        phi::EmptyLike<int64_t>(dev_ctx, ref_inputs[i]->indices()));
    output_values.emplace_back(
        phi::EmptyLike<T>(dev_ctx, ref_inputs[i]->values()));

    output_shapes.emplace_back(ref_inputs[i]->dims());
  }
  PADDLE_ENFORCE_EQ(
      input_tensor.nnz(),
      total_nnz,
      "The sum of all ref_inputs' nnz must equel input_tensor's nnz ");

  // input_tensor.indices().dim_size(0)
  // 看起应该是indices在第0个维度上的大小(也就是理论上对应densetensor的dim)

  for (int i = 0; i < num_split; ++i) {
    // TODO(ataei): Pass an allocator to avoid allocating large memory buffer.

    output_values_t.emplace_back(output_values[i].data<T>());
    output_indices_t.emplace_back(output_values[i].data<int64_t>());
  }
  std::vector<int64_t> slice_indexs;
  auto GetSliceIndex = [](int value, const std::vector<int>& intervals) -> int {
    for (size_t index = 0; index < intervals.size(); ++index) {
      if (value < intervals[index]) {
        // 可能不需要检查
        return static_cast<int>(index);
      }
    }
  };
  for (int64_t i = 0; i < input_tensor.indices().dims()[0]; ++i) {
    slice_indexs[i] = GetSliceIndex(i, slice_index_chowks);
  }
  // 看起来应该是保存
  std::vector<int> values_inserted_in_slice(num_split, 0);
  // 将对应的input_tensor 对应的indice分配到对应的输出上,同时划分对应的value
  int64_t input_nnz = input_tensor.indices().dims()[0];

  const int num_dim = input_tensor.indices().dims()[1];
  for (int i = 0; i < input_tensor.indices().dims()[0]; ++i) {
    // dim   表示在indice在划分域下对应的值
    const int dim = input_indices_t[i * out_nnz + split_dim];
    const int slice_index = slice_indexs[i];
    const int slice_dim = values_inserted_in_slice[slice_index]++;
    output_values_t[slice_index][slice_dim] = input_values_t[i];
    for (int j = 0; j < num_dim; ++j) {
      const int64_t original_dim = input_indices_t(i, j);
      // j == split_dim 的时候使用替换的indices
      output_indices_t[slice_index][slice_dim * num_values[i] + j] =
          (j == split_dim) ? GetDimensionInSlice(
                                 original_dim, slice_index, slice_index_chowks)
                           : original_dim;
    }

    // Helper for Split() that returns the dimension in the slice.
  }
  for (int i = 0; i < num_split; ++i) {
    result[i]->SetMember(output_indices[i], output_values[i], output_shapes[i]);
  }
}

template <typename T, typename Context>
void ConcatCooGradKernel(const Context& dev_ctx,
                         const std::vector<const SparseCooTensor*>& x,
                         const SparseCooTensor& out_grad,
                         const Scalar& axis_scalar,
                         std::vector<SparseCooTensor*> x_grad) {
  PADDLE_ENFORCE_NOT_NULL(
      x[0], phi::errors::NotFound("The first input tensor is not initalized."));

  auto axis = axis_scalar.to<int64_t>();
  axis = funcs::ComputeAxis(static_cast<int64_t>(axis),
                            static_cast<int64_t>(x[0]->dims().size()));
  // get output tensor that the name is not kEmptyVarName
  std::vector<DenseTensor*> outputs;
  for (auto& t : x_grad) {
    if (t && t->numel() != 0UL) {
      EmptyLikeCooKernel<T, Context>(dev_ctx, out_grad, t);
      outputs.push_back(outs[j]);
    } else {
      outputs.push_back(nullptr);
    }
  }

  phi::funcs::SplitFunctor<Context, T> split_functor;
  split_functor(dev_ctx, out_grad, x, static_cast<int>(axis), &outputs);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(concat_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ConcatCooGradKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
