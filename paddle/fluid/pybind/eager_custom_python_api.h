// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <iostream>

#include "paddle/common/ddim.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/eager/to_static/run_program_func.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/phi/core/enforce.h"

COMMON_DECLARE_bool(use_legacy_gemm);
using egr::ConvertAllInputsToDistTensor;
using egr::InputsContainDistTensor;

namespace paddle {
namespace pybind {

static PyObject *eager_api_linear(PyObject *self,
                                  PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto &x = GetTensorFromArgs("linear", "X", args, 0, false);
    auto &weight = GetTensorFromArgs("linear", "weight", args, 1, false);
    auto &bias = GetTensorFromArgs("linear", "Bias", args, 2, true);

    tstate = PyEval_SaveThread();

    if (bias.is_dist_tensor() || (bias.has_allocation() && bias.numel() > 0)) {
      const phi::distributed::ProcessMesh *mesh = nullptr;
      if (InputsContainDistTensor(&mesh, x, weight, bias)) {
        ConvertAllInputsToDistTensor(mesh, x, weight, bias);
      }
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11000 && \
     !(defined(_WIN32) || defined(WIN32)))
      if (!FLAGS_use_legacy_gemm) {
        // TODO(Pan Zhaowu): Add proper broadcast logic for batchsize unaligned
        // batch-gemm. Currently handles: (B..., k) x (k, n) -> (B..., n), with
        // 1D or scalar bias.

        // --- Original input tensor dimensions and values ---
        const auto &x_original_shape = x.shape();
        const size_t x_ndim_original = x_original_shape.size();

        const auto &weight_original_shape = weight.shape();
        const size_t weight_ndim_original = weight_original_shape.size();

        // Determine the 'k' and 'n' dimensions based on original shapes.
        // These values are crucial for potential 1D reshaping and output shape
        // calculation.
        const int64_t k_dim =
            x_original_shape[x_ndim_original - 1];  // Last dimension of X
        const int64_t n_dim =
            weight_original_shape[weight_ndim_original -
                                  1];  // Last dimension of Weight

        // --- Process 1D x and weight tensors by reshaping them to 2D if
        // necessary --- Subsequent operations will use these processed tensors.
        paddle::Tensor x_processed =
            x;  // Start with original, possibly reassign if reshaped
        paddle::Tensor weight_processed =
            weight;  // Start with original, possibly reassign if reshaped

        // If x is 1D (e.g., shape [k]), reshape it to [1, k] to fit the (B...,
        // k) x (k, n) pattern. This effectively treats a 1D vector as a row
        // vector for matrix multiplication.
        if (x_ndim_original == 1) {
          x_processed = reshape_ad_func(x, {1, k_dim});
        }
        // If weight is 1D (e.g., shape [n]), reshape it to [k, 1].
        // This implies 'n' was 1 in the original context, and 'k' is determined
        // by x. This effectively treats a 1D vector as a column vector. Note:
        // This 'else if' means if both x and weight are 1D, only x gets
        // reshaped currently. For (k) x (n) where n != k and both are 1D, the
        // semantics are ambiguous and not directly covered by (B..., k) x (k,
        // n). The current design implies weight is at least 2D or is treated as
        // [k, 1] if 1D.
        else if (weight_ndim_original == 1) {  // NOLINT
          weight_processed = reshape_ad_func(weight, {k_dim, 1});
        }

        // --- Recalculate dimensions based on processed tensors ---
        // These dimensions will be used for the actual GEMM operation.
        const auto &x_shape_current = x_processed.shape();
        const size_t x_ndim_current = x_shape_current.size();

        const auto &weight_shape_current = weight_processed.shape();
        const size_t weight_ndim_current = weight_shape_current.size();

        // Effective 'k' and 'n' for GEMM.
        const int64_t k_effective = x_shape_current[x_ndim_current - 1];
        const int64_t n_effective =
            weight_shape_current[weight_ndim_current - 1];

        // --- Determine the final output shape ---
        // Start with the processed x's shape, then modify the last dimension.
        std::vector<int64_t> output_shape_vec = x_shape_current;
        output_shape_vec[x_ndim_current - 1] = n_effective;

        // If the original x was 1D, the processed x became [1, k].
        // The output_shape_vec would be [1, n].
        // For 1D input, we usually expect a 1D output (shape [n]) if possible.
        if (x_ndim_original == 1 && output_shape_vec.size() > 1 &&
            output_shape_vec[0] == 1) {
          output_shape_vec.erase(
              output_shape_vec
                  .begin());  // Remove the artificial batch dimension
        }

        // Calculate the total number of elements in the batch dimensions of X.
        // This is used for reshaping X into a 2D matrix for addmm_ad_func.
        const int64_t x_batch_numel =
            std::accumulate(output_shape_vec.begin(),
                            output_shape_vec.end() - 1,
                            1LL,
                            std::multiplies<int64_t>());

        // --- Bias handling and GEMM execution ---
        // The condition now uses the processed weight's shape.
        // This branch typically handles (B..., k) x (k, n) where n > 1.
        if (weight_shape_current[0] > 1 && weight_shape_current[1] > 1) {
          paddle::Tensor bias_1d =
              bias;  // Create a mutable copy if modification is needed
          // Align bias' shape to 'n_effective'. If bias.numel() != n_effective,
          // tile it.
          if (bias.numel() != n_effective) {
            bias_1d = tile_ad_func(bias, {static_cast<int64_t>(n_effective)});
          }
          // Execute fused GEMM with epilogue.
          auto [out, _] = fused_gemm_epilogue_ad_func(
              x_processed, weight_processed, bias_1d, false, false, "none");

          // If original x was 1D and output_shape_vec is 1D (i.e., [n]),
          // but fused_gemm_epilogue_ad_func returns a 2D tensor ([1, n]),
          // reshape it back to the desired 1D output shape.
          if (x_ndim_original == 1 && out.shape().size() == 2 &&
              output_shape_vec.size() == 1) {
            out = reshape_ad_func(out, output_shape_vec);
          }

          PyEval_RestoreThread(tstate);
          tstate = nullptr;
          return ToPyObject(out);
        } else {
          // This branch handles cases where weight_processed is effectively 2D
          // with one dimension being 1, e.g., (B..., k) x (k, 1) resulting in
          // (B..., 1). Or when weight_processed was originally 1D and reshaped
          // to [k, 1].

          // Reshape bias to [1, n_effective] then tile to [x_batch_numel, 1]
          // for addmm_ad_func.
          paddle::Tensor bias_2d = tile_ad_func(
              reshape_ad_func(bias, {1, n_effective}), {x_batch_numel, 1});

          // Perform matrix multiplication using addmm_ad_func.
          // x_processed is reshaped to 2D [x_batch_numel, k_effective] for the
          // multiplication.
          auto out = addmm_ad_func(
              bias_2d,
              reshape_ad_func(x_processed, {x_batch_numel, k_effective}),
              weight_processed,
              1.0,
              1.0);

          // Reshape the final output to the target output_shape_vec.
          out = reshape_ad_func(out, output_shape_vec);

          PyEval_RestoreThread(tstate);
          tstate = nullptr;
          return ToPyObject(out);
        }
      } else  // NOLINT(readability/braces)
#endif
      {
        auto mm_out = matmul_ad_func(x, weight, false, false);
        auto out = add_ad_func(mm_out, bias);

        PyEval_RestoreThread(tstate);
        tstate = nullptr;
        return ToPyObject(out);
      }
    } else {
      const phi::distributed::ProcessMesh *mesh = nullptr;
      if (InputsContainDistTensor(&mesh, x, weight)) {
        ConvertAllInputsToDistTensor(mesh, x, weight);
      }

      auto mm_out = matmul_ad_func(x, weight, false, false);
      PyEval_RestoreThread(tstate);
      tstate = nullptr;
      return ToPyObject(mm_out);
    }
  } catch (paddle::platform::EnforceNotMet &exception) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    std::ostringstream sout;
    sout << exception.what();
    sout << "  [operator < linear > error]";
    exception.set_error_str(sout.str());
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_run_program(PyObject *self,
                                       PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X_info = GetPyArgumentInfo("run_program", "X", args, 0, true);
    TensorListBufferAllocator X_allocator(X_info.second);
    auto &X = GetTensorListFromArgsWithBuffer("run_program",
                                              "X",
                                              0,
                                              nullptr,
                                              X_info.first,
                                              X_info.second,
                                              X_allocator);

    auto Params_info =
        GetPyArgumentInfo("run_program", "Params", args, 1, true);
    TensorListBufferAllocator Params_allocator(Params_info.second);
    auto &Params = GetTensorListFromArgsWithBuffer("run_program",
                                                   "Params",
                                                   0,
                                                   nullptr,
                                                   Params_info.first,
                                                   Params_info.second,
                                                   Params_allocator);

    auto OutScope =
        GetScopePtrListFromArgs("run_program", "OutScope", args, 2, false);
    const phi::distributed::ProcessMesh *mesh = nullptr;
    if (InputsContainDistTensor(&mesh, X, Params)) {
      X = GetTensorListFromArgsWithBuffer("run_program",
                                          "X",
                                          0,
                                          nullptr,
                                          X_info.first,
                                          X_info.second,
                                          X_allocator);
      Params = GetTensorListFromArgsWithBuffer("run_program",
                                               "Params",
                                               0,
                                               nullptr,
                                               Params_info.first,
                                               Params_info.second,
                                               Params_allocator);
    }
    VLOG(6) << "Start PIR GetProgramAttributesMapPtrFromPyArgs";
    auto prog_attrs_ptr =
        GetProgramAttributesMapPtrFromPyArgs("run_program", args, 3);
    VLOG(6) << "Finish PIR GetProgramAttributesMapPtrFromPyArgs";

    VLOG(6) << "Start PIR ConstructCudaGraphAttrMapForRunProgram";
    paddle::framework::AttributeMap cuda_graph_attrs;
    ConstructCudaGraphAttrMapForRunProgram(
        "run_program", args, 4, cuda_graph_attrs);
    VLOG(6) << "Finish PIR ConstructCudaGraphAttrMapForRunProgram";
    tstate = PyEval_SaveThread();
    auto out = egr::to_static::run_program_ad_func(
        X, Params, OutScope, *prog_attrs_ptr, cuda_graph_attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (paddle::platform::EnforceNotMet &exception) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    std::ostringstream sout;
    sout << exception.what();
    sout << "  [operator < run_program > error]";
    exception.set_error_str(sout.str());
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyMethodDef CustomEagerFinalStateMethods[] = {
    {"linear",
     (PyCFunction)(void (*)(void))eager_api_linear,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for linear."},
    {"run_program",
     (PyCFunction)(void (*)(void))eager_api_run_program,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for run_program in dygraph."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace pybind
}  // namespace paddle
