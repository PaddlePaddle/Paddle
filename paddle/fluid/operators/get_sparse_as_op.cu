/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/get_sparse_as_op.h"
#include "paddle/fluid/platform/assert.h"

namespace paddle {
namespace operators {

template <typename T, int BlockDimX, int BlockDimY, int GridDimX,
          bool PaddingFlag>
__global__ void GetSparseAs(T *output, const T *table, const int64_t *ids,
                            const int64_t N, const int64_t K, const int64_t D,
                            const int64_t padding_idx) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int64_t id = ids[idy];
    PADDLE_ASSERT(id >= 0);
    PADDLE_ASSERT(id < N);
    T *out = output + idy * D;
    const T *tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      if (PaddingFlag) {
        if (id == padding_idx)
          out[i] = static_cast<T>(0);
        else
          out[i] = tab[i];
      } else {
        out[i] = tab[i];
      }
    }
    idy += BlockDimY * GridDimX;
  }
}

template <typename T>
class GetSparseAsCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    using LoDTensor = framework::LoDTensor;
    using SelectedRows = framework::SelectedRows;

    auto *w_t = context.Input<LoDTensor>("W");
    auto *x_s = context.Input<SelectedRows>("X");
    auto *out_s = context.Output<SelectedRows>("Out");

    out_s->set_height(x_s->height());
    out_s->set_rows(x_s->rows());
    out_s->mutable_value()->Resize(x_s->value().dims());

    auto *output_ptr =
        out_s->mutable_value()->mutable_data<T>(context.GetPlace());

    int64_t row_num = w_t->dims()[0];
    int64_t row_width = w_t->dims()[1];
    auto *table_ptr = w_t->data<T>();

    int64_t sparse_row_num = x_s->rows().size();
    auto ids_v = x_s->rows();

    dim3 threads(128, 8);
    dim3 grids(8, 1);

    GetSparseAs<T, 128, 8, 8, false /*padding_idx*/><<<
        grids, threads, 0, context.cuda_device_context().stream()>>>(
        output_ptr, table_ptr, ids_v.CUDAData(context.GetPlace()), row_num,
        sparse_row_num, row_width, -1 /*padding_idx*/);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(get_sparse_as, ops::GetSparseAsCUDAKernel<float>,
                        ops::GetSparseAsCUDAKernel<double>);
