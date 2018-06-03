/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/lookup_table_op.h"
#include "paddle/fluid/platform/assert.h"
#include "paddle/fluid/platform/cuda_primitives.h"

#define CLOG std::cout

namespace paddle {
namespace operators {

struct Formater {
  std::string message;
  std::string name;
  std::vector<int> dims;
  std::type_index dtype{typeid(const char)};
  framework::LoD lod;
  int summarize;
  void* data{nullptr};

  void operator()(size_t size) {
    // PrintMessage();
    // PrintName();
    // PrintDims();
    // PrintDtype();
    // PrintLod();
    PrintData(size);
  }

 private:
  void PrintMessage() { CLOG << std::time(nullptr) << "\t" << message << "\t"; }
  void PrintName() {
    if (!name.empty()) {
      CLOG << "Tensor[" << name << "]" << std::endl;
    }
  }
  void PrintDims() {
    if (!dims.empty()) {
      CLOG << "\tshape: [";
      for (auto i : dims) {
        CLOG << i << ",";
      }
      CLOG << "]" << std::endl;
    }
  }
  void PrintDtype() {
    if (dtype.hash_code() != typeid(const char).hash_code()) {
      CLOG << "\tdtype: " << dtype.name() << std::endl;
    }
  }
  void PrintLod() {
    if (!lod.empty()) {
      CLOG << "\tLoD: [";
      for (auto level : lod) {
        CLOG << "[ ";
        for (auto i : level) {
          CLOG << i << ",";
        }
        CLOG << " ]";
      }
      CLOG << "]" << std::endl;
    }
  }

  void PrintData(size_t size) {
    PADDLE_ENFORCE_NOT_NULL(data);
    // print float
    if (dtype.hash_code() == typeid(const float).hash_code()) {
      Display<float>(size);
    } else if (dtype.hash_code() == typeid(const double).hash_code()) {
      Display<double>(size);
    } else if (dtype.hash_code() == typeid(const int).hash_code()) {
      Display<int>(size);
    } else if (dtype.hash_code() == typeid(const int64_t).hash_code()) {
      Display<int64_t>(size);
    } else if (dtype.hash_code() == typeid(const bool).hash_code()) {
      Display<bool>(size);
    } else {
      CLOG << "\tdata: unprintable type: " << dtype.name() << std::endl;
    }
  }

  template <typename T>
  void Display(size_t size) {
    auto* d = reinterpret_cast<T*>(data);
    CLOG << "\tdata: " << size << std::endl;
    if (summarize != -1) {
      summarize = 10000;
      CLOG << "Value of summarize = " << summarize << std::endl;
      for (int i = 0; i < summarize; i++) {
        CLOG << d[i] << ",";
      }
    } else {
      for (size_t i = 0; i < size; i++) {
        CLOG << d[i] << ",";
      }
    }
    CLOG << std::endl;
  }
};

template <typename T, int BlockDimX, int BlockDimY, int GridDimX,
          bool PaddingFlag>
__global__ void LookupTable(T* output, const T* table, const int64_t* ids,
                            const int64_t N, const int64_t K, const int64_t D,
                            const int64_t padding_idx) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int64_t id = ids[idy];
    PADDLE_ASSERT(id >= 0);
    PADDLE_ASSERT(id < N);
    T* out = output + idy * D;
    const T* tab = table + id * D;
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

template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void LookupTableGrad(T* table, const T* output, const int64_t* ids,
                                const int64_t N, const int64_t K,
                                const int64_t D) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int id = ids[idy];
    PADDLE_ASSERT(id >= 0);
    PADDLE_ASSERT(id < N);
    const T* out = output + idy * D;
    T* tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      tab[i] = tab[i] + out[i];
      // paddle::platform::CudaAtomicAdd(&tab[i], out[i]);
    }
    idy += BlockDimY * GridDimX;
  }
}

template <typename T>
class LookupTableCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* table_t = context.Input<LoDTensor>("W");
    int64_t padding_idx = context.Attr<int64_t>("padding_idx");
    auto* ids_var = context.InputVar("Ids");
    Tensor* output_t = context.Output<Tensor>("Out");

    framework::LoDTensor print_tensor_table;
    print_tensor_table.set_lod(table_t->lod());
    print_tensor_table.Resize(table_t->dims());

    if (paddle::platform::is_cpu_place(table_t->place())) {
      print_tensor_table.ShareDataWith(*table_t);
    } else {
      // copy data to cpu to print
      paddle::platform::CPUPlace place;
      framework::TensorCopy(*table_t, place, &print_tensor_table);
    }
    Formater formater1;
    formater1.dtype = print_tensor_table.type();
    formater1.data = reinterpret_cast<void*>(print_tensor_table.data<void>());
    // formater1(print_tensor_table.numel());

    int64_t* ids;
    int64_t K;

    // The type of Ids(Input) is SelectedRows or LoDTensor, when Ids's type
    // is LoDTensor, this tensor contains the ids to be looked up in W;
    // when Ids's type is SelectedRows, the rows of Ids contains the
    // ids to be looked up in W.
    if (ids_var->IsType<framework::LoDTensor>()) {
      auto* ids_t = context.Input<LoDTensor>("Ids");
      ids = const_cast<int64_t*>(ids_t->data<int64_t>());
      K = ids_t->numel();

      framework::LoDTensor print_tensor_ids;
      print_tensor_ids.set_lod(ids_t->lod());
      print_tensor_ids.Resize(ids_t->dims());

      if (paddle::platform::is_cpu_place(ids_t->place())) {
        print_tensor_ids.ShareDataWith(*ids_t);
      } else {
        // copy data to cpu to print
        paddle::platform::CPUPlace place;
        framework::TensorCopy(*ids_t, place, &print_tensor_ids);
      }
      Formater formater2;
      formater2.dtype = print_tensor_ids.type();
      formater2.data = reinterpret_cast<void*>(print_tensor_ids.data<void>());
      // formater2(print_tensor_ids.numel());

    } else if (ids_var->IsType<framework::SelectedRows>()) {
      // std::cout << "Oh this is dealing with selected rows" << std::endl;
      auto* ids_t = context.Input<framework::SelectedRows>("Ids");
      ids = const_cast<int64_t*>(ids_t->rows().CUDAData(context.GetPlace()));
      K = ids_t->rows().size();
      output_t->Resize({K, table_t->dims()[1]});
    } else {
      PADDLE_THROW("Unsupported Variable Type of Ids");
    }

    size_t N = table_t->dims()[0];
    size_t D = table_t->dims()[1];
    auto* table = table_t->data<T>();
    auto* output = output_t->mutable_data<T>(context.GetPlace());

    dim3 threads(1, 1);
    dim3 grids(1, 1);

    if (padding_idx == -1)
      LookupTable<
          T, 1, 1, 1,
          false><<<grids, threads, 0, context.cuda_device_context().stream()>>>(
          output, table, ids, N, K, D, padding_idx);
    else
      LookupTable<
          T, 1, 1, 1,
          true><<<grids, threads, 0, context.cuda_device_context().stream()>>>(
          output, table, ids, N, K, D, padding_idx);

    framework::LoDTensor print_tensor_output;
    print_tensor_output.Resize(output_t->dims());
    // std::cout << print_tensor_output.dims() << std::endl;

    if (paddle::platform::is_cpu_place(output_t->place())) {
      print_tensor_table.ShareDataWith(*output_t);
    } else {
      // copy data to cpu to print
      paddle::platform::CPUPlace place;
      framework::TensorCopy(*output_t, place, &print_tensor_output);
    }
    Formater formater3;
    formater3.dtype = print_tensor_output.type();
    formater3.data = reinterpret_cast<void*>(print_tensor_output.data<void>());
    // formater3(print_tensor_output.numel());
  }
};

template <typename T>
class LookupTableGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // std::cout << "Now we are in the backward kernel" << std::endl;
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    bool is_sparse = context.Attr<bool>("is_sparse");
    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    if (is_sparse) {
      // std::cout << "Is_sparse is true" << std::endl;
      auto* ids = context.Input<LoDTensor>("Ids");
      auto* table = context.Input<LoDTensor>("W");
      auto* d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto* d_table = context.Output<SelectedRows>(framework::GradVarName("W"));

      auto* ids_data = ids->data<int64_t>();
      auto ids_dim = ids->dims();

      auto stream = dev_ctx.stream();
      // copy GPU memory to CPU pinned memory
      framework::Vector<int64_t> new_rows;
      new_rows.resize(ids_dim[0]);
      auto gpu_place = boost::get<platform::CUDAPlace>(context.GetPlace());

      // TODO(yuyang18): Strange code here.
      memory::Copy(platform::CPUPlace(),
                   new_rows.CUDAMutableData(context.GetPlace()), gpu_place,
                   ids_data, ids_dim[0] * sizeof(int64_t), stream);

      d_table->set_rows(new_rows);

      auto* d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_dim[0], table->dims()[1]});
      d_table_value->mutable_data<T>(context.GetPlace());

      auto* d_table_data = d_table_value->data<T>();
      auto* d_output_data = d_output->data<T>();
      PADDLE_ENFORCE_EQ(d_table_value->dims(), d_output->dims());
      memory::Copy(gpu_place, d_table_data, gpu_place, d_output_data,
                   d_output->numel() * sizeof(T), stream);

      framework::LoDTensor print_tensor_table;
      print_tensor_table.Resize(table->dims());
      // std::cout << "Printing table W" << std::endl;
      // std::cout << print_tensor_table.dims() << std::endl;

      if (paddle::platform::is_cpu_place(table->place())) {
        print_tensor_table.ShareDataWith(*table);
      } else {
        // copy data to cpu to print
        // std::cout << "Should be printed" << std::endl;
        paddle::platform::CPUPlace place;
        framework::TensorCopy(*table, place, &print_tensor_table);
      }
      Formater formater3;
      formater3.dtype = print_tensor_table.type();
      formater3.data = reinterpret_cast<void*>(print_tensor_table.data<void>());
      // formater3(print_tensor_table.numel());

      // Printing doutput
      framework::LoDTensor print_tensor_doutput;
      print_tensor_doutput.Resize(d_output->dims());
      // std::cout << "Printing d_output" << std::endl;
      // std::cout << print_tensor_doutput.dims() << std::endl;

      if (paddle::platform::is_cpu_place(d_output->place())) {
        print_tensor_doutput.ShareDataWith(*d_output);
      } else {
        // copy data to cpu to print
        paddle::platform::CPUPlace place;
        framework::TensorCopy(*d_output, place, &print_tensor_doutput);
      }
      Formater formater5;
      formater5.dtype = print_tensor_doutput.type();
      formater5.data =
          reinterpret_cast<void*>(print_tensor_doutput.data<void>());
      // formater5(print_tensor_doutput.numel());

    } else {
      // std::cout << "Is_sparse is false, hence we are here" << std::endl;
      auto ids_t = context.Input<LoDTensor>("Ids");
      auto d_output_t = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto d_table_t = context.Output<LoDTensor>(framework::GradVarName("W"));

      int N = d_table_t->dims()[0];
      int D = d_table_t->dims()[1];
      int K = ids_t->numel();
      const int64_t* ids = ids_t->data<int64_t>();
      const T* d_output = d_output_t->data<T>();
      T* d_table = d_table_t->mutable_data<T>(context.GetPlace());

      auto t = framework::EigenVector<T>::Flatten(*d_table_t);
      t.device(*dev_ctx.eigen_device()) = t.constant(static_cast<T>(0));

      dim3 threads(1, 1);
      dim3 grids(1, 1);
      LookupTableGrad<T, 1, 1, 1><<<grids, threads, 0, dev_ctx.stream()>>>(
          d_table, d_output, ids, N, K, D);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(lookup_table, ops::LookupTableCUDAKernel<float>,
                        ops::LookupTableCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(lookup_table_grad,
                        ops::LookupTableGradCUDAKernel<float>,
                        ops::LookupTableGradCUDAKernel<double>);
