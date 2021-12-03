// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <Python.h>
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "pybind11/detail/common.h"

namespace paddle {
namespace pybind {

static PyObject *eager_api_rsqrt(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("rsqrt", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("rsqrt", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = rsqrt_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_multihead_matmul(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("multihead_matmul", "Input", args, 0, false);
    auto W = GetEagerTensorFromArgs("multihead_matmul", "W", args, 1, false);
    auto Bias =
        GetEagerTensorFromArgs("multihead_matmul", "Bias", args, 2, false);
    auto BiasQK =
        GetEagerTensorFromArgs("multihead_matmul", "BiasQK", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("multihead_matmul", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = multihead_matmul_dygraph_function(Input, W, Bias, BiasQK, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_addmm(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("addmm", "Input", args, 0, false);
    auto X = GetEagerTensorFromArgs("addmm", "X", args, 1, false);
    auto Y = GetEagerTensorFromArgs("addmm", "Y", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("addmm", args, 3, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = addmm_dygraph_function(Input, X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_gru(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("gru", "Input", args, 0, false);
    auto Weight = GetEagerTensorFromArgs("gru", "Weight", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("gru", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = gru_dygraph_function(Input, Weight, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_round(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("round", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("round", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = round_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_rank_attention(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("rank_attention", "X", args, 0, false);
    auto RankOffset =
        GetEagerTensorFromArgs("rank_attention", "RankOffset", args, 1, false);
    auto RankParam =
        GetEagerTensorFromArgs("rank_attention", "RankParam", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("rank_attention", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = rank_attention_dygraph_function(X, RankOffset, RankParam, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_embedding_fc_lstm(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Ids = GetEagerTensorFromArgs("fused_embedding_fc_lstm", "Ids", args, 0,
                                      false);
    auto Embeddings = GetEagerTensorFromArgs("fused_embedding_fc_lstm",
                                             "Embeddings", args, 1, false);
    auto WeightH = GetEagerTensorFromArgs("fused_embedding_fc_lstm", "WeightH",
                                          args, 2, false);
    auto Bias = GetEagerTensorFromArgs("fused_embedding_fc_lstm", "Bias", args,
                                       3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_embedding_fc_lstm", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_embedding_fc_lstm_dygraph_function(Ids, Embeddings,
                                                        WeightH, Bias, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_where_index(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Condition =
        GetEagerTensorFromArgs("where_index", "Condition", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("where_index", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = where_index_dygraph_function(Condition, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bicubic_interp(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bicubic_interp", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bicubic_interp", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = bicubic_interp_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_arg_min(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("arg_min", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("arg_min", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = arg_min_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_tile(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("tile", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("tile", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = tile_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bilinear_tensor_product(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("bilinear_tensor_product", "X", args, 0, false);
    auto Y =
        GetEagerTensorFromArgs("bilinear_tensor_product", "Y", args, 1, false);
    auto Weight = GetEagerTensorFromArgs("bilinear_tensor_product", "Weight",
                                         args, 2, false);
    auto Bias = GetEagerTensorFromArgs("bilinear_tensor_product", "Bias", args,
                                       3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bilinear_tensor_product", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = bilinear_tensor_product_dygraph_function(X, Y, Weight, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_ctc_align(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("ctc_align", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("ctc_align", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = ctc_align_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pow2_decay_with_linear_warmup(PyObject *self,
                                                         PyObject *args,
                                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto LearningRate = GetEagerTensorFromArgs("pow2_decay_with_linear_warmup",
                                               "LearningRate", args, 0, false);
    auto Step = GetEagerTensorFromArgs("pow2_decay_with_linear_warmup", "Step",
                                       args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pow2_decay_with_linear_warmup", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = pow2_decay_with_linear_warmup_dygraph_function(LearningRate,
                                                              Step, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_split(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("split", "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs("split", "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("split", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = split_dygraph_function(X, OutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fc(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("fc", "Input", args, 0, false);
    auto W = GetEagerTensorFromArgs("fc", "W", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fc", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fc_dygraph_function(Input, W, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_clear_float_status(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto FloatStatus = GetEagerTensorFromArgs("clear_float_status",
                                              "FloatStatus", args, 0, false);
    auto FloatStatusOut = GetEagerTensorFromArgs(
        "clear_float_status", "FloatStatusOut", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("clear_float_status", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = clear_float_status_dygraph_function(FloatStatus, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_matmul_v2(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("matmul_v2", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("matmul_v2", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("matmul_v2", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = matmul_v2_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_load(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("load", args, 0, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = load_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elementwise_max(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("elementwise_max", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("elementwise_max", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elementwise_max", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elementwise_max_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_adadelta(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("adadelta", "Param", args, 0, false);
    auto Grad = GetEagerTensorFromArgs("adadelta", "Grad", args, 1, false);
    auto AvgSquaredGrad =
        GetEagerTensorFromArgs("adadelta", "AvgSquaredGrad", args, 2, false);
    auto AvgSquaredUpdate =
        GetEagerTensorFromArgs("adadelta", "AvgSquaredUpdate", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("adadelta", args, 4, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = adadelta_dygraph_function(Param, Grad, AvgSquaredGrad,
                                         AvgSquaredUpdate, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_chunk_eval(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Inference =
        GetEagerTensorFromArgs("chunk_eval", "Inference", args, 0, false);
    auto Label = GetEagerTensorFromArgs("chunk_eval", "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("chunk_eval", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = chunk_eval_dygraph_function(Inference, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_check_finite_and_unscale(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("check_finite_and_unscale", "X", args,
                                        0, false);
    auto Scale = GetEagerTensorFromArgs("check_finite_and_unscale", "Scale",
                                        args, 1, false);
    auto Out = GetEagerTensorListFromArgs("check_finite_and_unscale", "Out",
                                          args, 2, false);
    auto FoundInfinite = GetEagerTensorFromArgs(
        "check_finite_and_unscale", "FoundInfinite", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("check_finite_and_unscale", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = check_finite_and_unscale_dygraph_function(X, Scale, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sparse_momentum(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param =
        GetEagerTensorFromArgs("sparse_momentum", "Param", args, 0, false);
    auto Grad =
        GetEagerTensorFromArgs("sparse_momentum", "Grad", args, 1, false);
    auto Velocity =
        GetEagerTensorFromArgs("sparse_momentum", "Velocity", args, 2, false);
    auto Index =
        GetEagerTensorFromArgs("sparse_momentum", "Index", args, 3, false);
    auto LearningRate = GetEagerTensorFromArgs("sparse_momentum",
                                               "LearningRate", args, 4, false);
    auto ParamOut =
        GetEagerTensorFromArgs("sparse_momentum", "ParamOut", args, 5, false);
    auto VelocityOut = GetEagerTensorFromArgs("sparse_momentum", "VelocityOut",
                                              args, 6, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sparse_momentum", args, 7,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sparse_momentum_dygraph_function(Param, Grad, Velocity, Index,
                                                LearningRate, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_tan(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("tan", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("tan", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = tan_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_adam(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("adam", "Param", args, 0, false);
    auto Grad = GetEagerTensorFromArgs("adam", "Grad", args, 1, false);
    auto LearningRate =
        GetEagerTensorFromArgs("adam", "LearningRate", args, 2, false);
    auto Moment1 = GetEagerTensorFromArgs("adam", "Moment1", args, 3, false);
    auto Moment2 = GetEagerTensorFromArgs("adam", "Moment2", args, 4, false);
    auto Beta1Pow = GetEagerTensorFromArgs("adam", "Beta1Pow", args, 5, false);
    auto Beta2Pow = GetEagerTensorFromArgs("adam", "Beta2Pow", args, 6, false);
    auto MasterParam =
        GetEagerTensorFromArgs("adam", "MasterParam", args, 7, true);
    auto ParamOut = GetEagerTensorFromArgs("adam", "ParamOut", args, 8, false);
    auto Moment1Out =
        GetEagerTensorFromArgs("adam", "Moment1Out", args, 9, false);
    auto Moment2Out =
        GetEagerTensorFromArgs("adam", "Moment2Out", args, 10, false);
    auto Beta1PowOut =
        GetEagerTensorFromArgs("adam", "Beta1PowOut", args, 11, false);
    auto Beta2PowOut =
        GetEagerTensorFromArgs("adam", "Beta2PowOut", args, 12, false);
    auto MasterParamOut =
        GetEagerTensorFromArgs("adam", "MasterParamOut", args, 13, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("adam", args, 14, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = adam_dygraph_function(Param, Grad, LearningRate, Moment1,
                                     Moment2, Beta1Pow, Beta2Pow, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fsp(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fsp", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("fsp", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fsp", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fsp_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_where(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Condition =
        GetEagerTensorFromArgs("where", "Condition", args, 0, false);
    auto X = GetEagerTensorFromArgs("where", "X", args, 1, false);
    auto Y = GetEagerTensorFromArgs("where", "Y", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("where", args, 3, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = where_dygraph_function(Condition, X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_logical_xor(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("logical_xor", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("logical_xor", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("logical_xor", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = logical_xor_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_multiclass_nms3(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto BBoxes =
        GetEagerTensorFromArgs("multiclass_nms3", "BBoxes", args, 0, false);
    auto Scores =
        GetEagerTensorFromArgs("multiclass_nms3", "Scores", args, 1, false);
    auto RoisNum =
        GetEagerTensorFromArgs("multiclass_nms3", "RoisNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("multiclass_nms3", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = multiclass_nms3_dygraph_function(BBoxes, Scores, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_one_hot_v2(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("one_hot_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("one_hot_v2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = one_hot_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_softmax(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_softmax", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_softmax", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_softmax_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_affine_channel(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("affine_channel", "X", args, 0, false);
    auto Scale =
        GetEagerTensorFromArgs("affine_channel", "Scale", args, 1, false);
    auto Bias =
        GetEagerTensorFromArgs("affine_channel", "Bias", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("affine_channel", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = affine_channel_dygraph_function(X, Scale, Bias, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_triangular_solve(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("triangular_solve", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("triangular_solve", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("triangular_solve", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = triangular_solve_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_topk_avg_pooling(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_topk_avg_pooling", "X", args, 0,
                                    false);
    auto ROW = GetEagerTensorFromArgs("sequence_topk_avg_pooling", "ROW", args,
                                      1, false);
    auto COLUMN = GetEagerTensorFromArgs("sequence_topk_avg_pooling", "COLUMN",
                                         args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_topk_avg_pooling", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        sequence_topk_avg_pooling_dygraph_function(X, ROW, COLUMN, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_space_to_depth(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("space_to_depth", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("space_to_depth", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = space_to_depth_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reverse(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reverse", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reverse", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reverse_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_embedding_eltwise_layernorm(PyObject *self,
                                                             PyObject *args,
                                                             PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Ids = GetEagerTensorListFromArgs("fused_embedding_eltwise_layernorm",
                                          "Ids", args, 0, false);
    auto Embs = GetEagerTensorListFromArgs("fused_embedding_eltwise_layernorm",
                                           "Embs", args, 1, false);
    auto Bias = GetEagerTensorFromArgs("fused_embedding_eltwise_layernorm",
                                       "Bias", args, 2, false);
    auto Scale = GetEagerTensorFromArgs("fused_embedding_eltwise_layernorm",
                                        "Scale", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_embedding_eltwise_layernorm", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_embedding_eltwise_layernorm_dygraph_function(
        Ids, Embs, Bias, Scale, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_expand_v2(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("expand_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("expand_v2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = expand_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lgamma(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("lgamma", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lgamma", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = lgamma_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_solve(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("solve", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("solve", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("solve", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = solve_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_deformable_psroi_pooling(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("deformable_psroi_pooling", "Input",
                                        args, 0, false);
    auto ROIs = GetEagerTensorFromArgs("deformable_psroi_pooling", "ROIs", args,
                                       1, false);
    auto Trans = GetEagerTensorFromArgs("deformable_psroi_pooling", "Trans",
                                        args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("deformable_psroi_pooling", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        deformable_psroi_pooling_dygraph_function(Input, ROIs, Trans, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_instance_norm(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("instance_norm", "X", args, 0, false);
    auto Scale =
        GetEagerTensorFromArgs("instance_norm", "Scale", args, 1, true);
    auto Bias = GetEagerTensorFromArgs("instance_norm", "Bias", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("instance_norm", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = instance_norm_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_decode_jpeg(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("decode_jpeg", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("decode_jpeg", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = decode_jpeg_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_gather_nd(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("gather_nd", "X", args, 0, false);
    auto Index = GetEagerTensorFromArgs("gather_nd", "Index", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("gather_nd", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = gather_nd_dygraph_function(X, Index, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reduce_prod(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reduce_prod", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reduce_prod", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reduce_prod_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_matrix_rank(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("matrix_rank", "X", args, 0, false);
    auto TolTensor =
        GetEagerTensorFromArgs("matrix_rank", "TolTensor", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("matrix_rank", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = matrix_rank_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_asin(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("asin", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("asin", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = asin_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lstmp(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("lstmp", "Input", args, 0, false);
    auto Weight = GetEagerTensorFromArgs("lstmp", "Weight", args, 1, false);
    auto ProjWeight =
        GetEagerTensorFromArgs("lstmp", "ProjWeight", args, 2, false);
    auto Bias = GetEagerTensorFromArgs("lstmp", "Bias", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lstmp", args, 4, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = lstmp_dygraph_function(Input, Weight, ProjWeight, Bias, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_iou_similarity(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("iou_similarity", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("iou_similarity", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("iou_similarity", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = iou_similarity_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_huber_loss(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("huber_loss", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("huber_loss", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("huber_loss", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = huber_loss_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_one_hot(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("one_hot", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("one_hot", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = one_hot_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_slice(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_slice", "X", args, 0, false);
    auto Offset =
        GetEagerTensorFromArgs("sequence_slice", "Offset", args, 1, false);
    auto Length =
        GetEagerTensorFromArgs("sequence_slice", "Length", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_slice", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_slice_dygraph_function(X, Offset, Length, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lookup_table(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto W = GetEagerTensorFromArgs("lookup_table", "W", args, 0, false);
    auto Ids = GetEagerTensorFromArgs("lookup_table", "Ids", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lookup_table", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = lookup_table_dygraph_function(W, Ids, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_softplus(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("softplus", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("softplus", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = softplus_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_depthwise_conv2d(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("depthwise_conv2d", "Input", args, 0, false);
    auto Filter =
        GetEagerTensorFromArgs("depthwise_conv2d", "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("depthwise_conv2d", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = depthwise_conv2d_dygraph_function(Input, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_fc_elementwise_layernorm(PyObject *self,
                                                          PyObject *args,
                                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fused_fc_elementwise_layernorm", "X", args,
                                    0, false);
    auto W = GetEagerTensorFromArgs("fused_fc_elementwise_layernorm", "W", args,
                                    1, false);
    auto Y = GetEagerTensorFromArgs("fused_fc_elementwise_layernorm", "Y", args,
                                    2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_fc_elementwise_layernorm", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_fc_elementwise_layernorm_dygraph_function(X, W, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sigmoid_cross_entropy_with_logits(PyObject *self,
                                                             PyObject *args,
                                                             PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sigmoid_cross_entropy_with_logits", "X",
                                    args, 0, false);
    auto Label = GetEagerTensorFromArgs("sigmoid_cross_entropy_with_logits",
                                        "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sigmoid_cross_entropy_with_logits", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        sigmoid_cross_entropy_with_logits_dygraph_function(X, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_exp(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("exp", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("exp", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = exp_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_scatter(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("scatter", "X", args, 0, false);
    auto Ids = GetEagerTensorFromArgs("scatter", "Ids", args, 1, false);
    auto Updates = GetEagerTensorFromArgs("scatter", "Updates", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("scatter", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = scatter_dygraph_function(X, Ids, Updates, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_equal_all(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("equal_all", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("equal_all", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("equal_all", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = equal_all_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_searchsorted(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto SortedSequence = GetEagerTensorFromArgs(
        "searchsorted", "SortedSequence", args, 0, false);
    auto Values =
        GetEagerTensorFromArgs("searchsorted", "Values", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("searchsorted", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = searchsorted_dygraph_function(SortedSequence, Values, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fusion_squared_mat_sub(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("fusion_squared_mat_sub", "X", args, 0, false);
    auto Y =
        GetEagerTensorFromArgs("fusion_squared_mat_sub", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fusion_squared_mat_sub", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fusion_squared_mat_sub_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_unique(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("unique", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("unique", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = unique_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_log(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("log", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("log", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = log_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_conv_shift(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("conv_shift", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("conv_shift", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("conv_shift", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = conv_shift_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_smooth_l1_loss(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("smooth_l1_loss", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("smooth_l1_loss", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("smooth_l1_loss", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = smooth_l1_loss_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_linear_interp_v2(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("linear_interp_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("linear_interp_v2", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = linear_interp_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_momentum(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("momentum", "Param", args, 0, false);
    auto Grad = GetEagerTensorFromArgs("momentum", "Grad", args, 1, false);
    auto Velocity =
        GetEagerTensorFromArgs("momentum", "Velocity", args, 2, false);
    auto LearningRate =
        GetEagerTensorFromArgs("momentum", "LearningRate", args, 3, false);
    auto MasterParam =
        GetEagerTensorFromArgs("momentum", "MasterParam", args, 4, true);
    auto ParamOut =
        GetEagerTensorFromArgs("momentum", "ParamOut", args, 5, false);
    auto VelocityOut =
        GetEagerTensorFromArgs("momentum", "VelocityOut", args, 6, false);
    auto MasterParamOut =
        GetEagerTensorFromArgs("momentum", "MasterParamOut", args, 7, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("momentum", args, 8, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out =
        momentum_dygraph_function(Param, Grad, Velocity, LearningRate, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_temporal_shift(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("temporal_shift", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("temporal_shift", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = temporal_shift_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_nce(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("nce", "Input", args, 0, false);
    auto Label = GetEagerTensorFromArgs("nce", "Label", args, 1, false);
    auto Weight = GetEagerTensorFromArgs("nce", "Weight", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("nce", args, 3, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = nce_dygraph_function(Input, Label, Weight, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_mv(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("mv", "X", args, 0, false);
    auto Vec = GetEagerTensorFromArgs("mv", "Vec", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("mv", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = mv_dygraph_function(X, Vec, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_proximal_gd(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("proximal_gd", "Param", args, 0, false);
    auto Grad = GetEagerTensorFromArgs("proximal_gd", "Grad", args, 1, false);
    auto LearningRate =
        GetEagerTensorFromArgs("proximal_gd", "LearningRate", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("proximal_gd", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = proximal_gd_dygraph_function(Param, Grad, LearningRate, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_memcpy_h2d(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("memcpy_h2d", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("memcpy_h2d", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = memcpy_h2d_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_add_position_encoding(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("add_position_encoding", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("add_position_encoding", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = add_position_encoding_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cosh(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cosh", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cosh", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = cosh_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_hash(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("hash", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("hash", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = hash_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_grad_add(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("grad_add", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("grad_add", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("grad_add", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = grad_add_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sign(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sign", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sign", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sign_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_prelu(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("prelu", "X", args, 0, false);
    auto Alpha = GetEagerTensorFromArgs("prelu", "Alpha", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("prelu", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = prelu_dygraph_function(X, Alpha, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_linspace(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Start = GetEagerTensorFromArgs("linspace", "Start", args, 0, false);
    auto Stop = GetEagerTensorFromArgs("linspace", "Stop", args, 1, false);
    auto Num = GetEagerTensorFromArgs("linspace", "Num", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("linspace", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = linspace_dygraph_function(Start, Stop, Num, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fill_diagonal(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fill_diagonal", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fill_diagonal", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fill_diagonal_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_logsigmoid(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("logsigmoid", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("logsigmoid", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = logsigmoid_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_load_combine(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto OutNum =
        GetUnsignedLongFromArgs("load_combine", "OutNum", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("load_combine", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = load_combine_dygraph_function(OutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fetch_v2(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fetch_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fetch_v2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fetch_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_randperm(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("randperm", args, 0, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = randperm_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_scatter(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_scatter", "X", args, 0, false);
    auto Ids =
        GetEagerTensorFromArgs("sequence_scatter", "Ids", args, 1, false);
    auto Updates =
        GetEagerTensorFromArgs("sequence_scatter", "Updates", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_scatter", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_scatter_dygraph_function(X, Ids, Updates, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_partial_sum(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("partial_sum", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("partial_sum", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = partial_sum_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_relu6(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("relu6", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("relu6", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = relu6_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_conv3d(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("conv3d", "Input", args, 0, false);
    auto Filter = GetEagerTensorFromArgs("conv3d", "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("conv3d", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = conv3d_dygraph_function(Input, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lstm_unit(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("lstm_unit", "X", args, 0, false);
    auto C_prev = GetEagerTensorFromArgs("lstm_unit", "C_prev", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lstm_unit", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = lstm_unit_dygraph_function(X, C_prev, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_not_equal(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("not_equal", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("not_equal", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("not_equal", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = not_equal_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_transpose2(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("transpose2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("transpose2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = transpose2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_uniform_random_batch_size_like(PyObject *self,
                                                          PyObject *args,
                                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("uniform_random_batch_size_like",
                                        "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("uniform_random_batch_size_like", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = uniform_random_batch_size_like_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_unfold(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("unfold", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("unfold", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = unfold_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lrn(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("lrn", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lrn", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = lrn_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_softmax_with_cross_entropy(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Logits = GetEagerTensorFromArgs("softmax_with_cross_entropy", "Logits",
                                         args, 0, false);
    auto Label = GetEagerTensorFromArgs("softmax_with_cross_entropy", "Label",
                                        args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("softmax_with_cross_entropy", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        softmax_with_cross_entropy_dygraph_function(Logits, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_isfinite_v2(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("isfinite_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("isfinite_v2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = isfinite_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bernoulli(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bernoulli", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bernoulli", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = bernoulli_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_max_pool3d_with_index(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("max_pool3d_with_index", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("max_pool3d_with_index", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = max_pool3d_with_index_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_gaussian_random(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("gaussian_random", args, 0,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = gaussian_random_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_flatten2(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("flatten2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("flatten2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = flatten2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_matmul(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("matmul", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("matmul", "Y", args, 1, false);
    auto Out = GetEagerTensorFromArgs("matmul", "Out", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("matmul", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = matmul_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cvm(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cvm", "X", args, 0, false);
    auto CVM = GetEagerTensorFromArgs("cvm", "CVM", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cvm", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = cvm_dygraph_function(X, CVM, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_adamax(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("adamax", "Param", args, 0, false);
    auto Grad = GetEagerTensorFromArgs("adamax", "Grad", args, 1, false);
    auto LearningRate =
        GetEagerTensorFromArgs("adamax", "LearningRate", args, 2, false);
    auto Moment = GetEagerTensorFromArgs("adamax", "Moment", args, 3, false);
    auto InfNorm = GetEagerTensorFromArgs("adamax", "InfNorm", args, 4, false);
    auto Beta1Pow =
        GetEagerTensorFromArgs("adamax", "Beta1Pow", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("adamax", args, 6, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = adamax_dygraph_function(Param, Grad, LearningRate, Moment,
                                       InfNorm, Beta1Pow, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_masked_select(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("masked_select", "X", args, 0, false);
    auto Mask = GetEagerTensorFromArgs("masked_select", "Mask", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("masked_select", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = masked_select_dygraph_function(X, Mask, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_range(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Start = GetEagerTensorFromArgs("range", "Start", args, 0, false);
    auto End = GetEagerTensorFromArgs("range", "End", args, 1, false);
    auto Step = GetEagerTensorFromArgs("range", "Step", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("range", args, 3, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = range_dygraph_function(Start, End, Step, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bitwise_not(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bitwise_not", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bitwise_not", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = bitwise_not_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_trace(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("trace", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("trace", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = trace_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_multinomial(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("multinomial", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("multinomial", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = multinomial_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_modified_huber_loss(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("modified_huber_loss", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("modified_huber_loss", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("modified_huber_loss", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = modified_huber_loss_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_roll(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("roll", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("roll", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = roll_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_squared_l2_distance(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("squared_l2_distance", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("squared_l2_distance", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("squared_l2_distance", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = squared_l2_distance_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_conv3d_transpose(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("conv3d_transpose", "Input", args, 0, false);
    auto Filter =
        GetEagerTensorFromArgs("conv3d_transpose", "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("conv3d_transpose", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = conv3d_transpose_dygraph_function(Input, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_share_data(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("share_data", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("share_data", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = share_data_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fake_quantize_abs_max(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("fake_quantize_abs_max", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fake_quantize_abs_max", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fake_quantize_abs_max_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_unique_with_counts(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("unique_with_counts", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("unique_with_counts", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = unique_with_counts_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fill(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fill", args, 0, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fill_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_concat(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("concat", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("concat", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = concat_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fill_zeros_like(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fill_zeros_like", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fill_zeros_like", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fill_zeros_like_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_hierarchical_sigmoid(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("hierarchical_sigmoid", "X", args, 0, false);
    auto W =
        GetEagerTensorFromArgs("hierarchical_sigmoid", "W", args, 1, false);
    auto Label =
        GetEagerTensorFromArgs("hierarchical_sigmoid", "Label", args, 2, false);
    auto PathTable = GetEagerTensorFromArgs("hierarchical_sigmoid", "PathTable",
                                            args, 3, true);
    auto PathCode = GetEagerTensorFromArgs("hierarchical_sigmoid", "PathCode",
                                           args, 4, true);
    auto Bias =
        GetEagerTensorFromArgs("hierarchical_sigmoid", "Bias", args, 5, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("hierarchical_sigmoid", args, 6,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = hierarchical_sigmoid_dygraph_function(X, W, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_isinf_v2(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("isinf_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("isinf_v2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = isinf_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_squeeze(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("squeeze", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("squeeze", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = squeeze_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_multiclass_nms2(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto BBoxes =
        GetEagerTensorFromArgs("multiclass_nms2", "BBoxes", args, 0, false);
    auto Scores =
        GetEagerTensorFromArgs("multiclass_nms2", "Scores", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("multiclass_nms2", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = multiclass_nms2_dygraph_function(BBoxes, Scores, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bpr_loss(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bpr_loss", "X", args, 0, false);
    auto Label = GetEagerTensorFromArgs("bpr_loss", "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bpr_loss", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = bpr_loss_dygraph_function(X, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fft_c2c(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fft_c2c", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fft_c2c", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fft_c2c_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bicubic_interp_v2(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bicubic_interp_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bicubic_interp_v2", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = bicubic_interp_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reshape(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reshape", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reshape", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reshape_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_coalesce_tensor(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorListFromArgs("coalesce_tensor", "Input", args, 0, false);
    auto OutputNum =
        GetUnsignedLongFromArgs("coalesce_tensor", "OutputNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("coalesce_tensor", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = coalesce_tensor_dygraph_function(Input, OutputNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_roi_align(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("roi_align", "X", args, 0, false);
    auto ROIs = GetEagerTensorFromArgs("roi_align", "ROIs", args, 1, false);
    auto RoisNum =
        GetEagerTensorFromArgs("roi_align", "RoisNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("roi_align", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = roi_align_dygraph_function(X, ROIs, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reshape2(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reshape2", "X", args, 0, false);
    auto Shape = GetEagerTensorFromArgs("reshape2", "Shape", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reshape2", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reshape2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reduce_any(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reduce_any", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reduce_any", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reduce_any_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_unstack(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("unstack", "X", args, 0, false);
    auto YNum = GetUnsignedLongFromArgs("unstack", "YNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("unstack", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = unstack_dygraph_function(X, YNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_scatter_nd_add(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("scatter_nd_add", "X", args, 0, false);
    auto Index =
        GetEagerTensorFromArgs("scatter_nd_add", "Index", args, 1, false);
    auto Updates =
        GetEagerTensorFromArgs("scatter_nd_add", "Updates", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("scatter_nd_add", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = scatter_nd_add_dygraph_function(X, Index, Updates, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_reshape(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_reshape", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_reshape", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_reshape_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bilateral_slice(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bilateral_slice", "X", args, 0, false);
    auto Grid =
        GetEagerTensorFromArgs("bilateral_slice", "Grid", args, 1, false);
    auto Guide =
        GetEagerTensorFromArgs("bilateral_slice", "Guide", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bilateral_slice", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = bilateral_slice_dygraph_function(X, Grid, Guide, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fill_any_like(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fill_any_like", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fill_any_like", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fill_any_like_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_empty(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("empty", args, 0, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = empty_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pad_constant_like(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("pad_constant_like", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("pad_constant_like", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pad_constant_like", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = pad_constant_like_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pool2d(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("pool2d", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pool2d", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = pool2d_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_size(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("size", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("size", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = size_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_imag(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("imag", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("imag", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = imag_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_eigh(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("eigh", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("eigh", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = eigh_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_stack(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("stack", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("stack", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = stack_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_dgc_momentum(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto current_step =
        GetEagerTensorFromArgs("dgc_momentum", "current_step", args, 0, false);
    auto nranks =
        GetEagerTensorFromArgs("dgc_momentum", "nranks", args, 1, false);
    auto Param =
        GetEagerTensorFromArgs("dgc_momentum", "Param", args, 2, false);
    auto Grad = GetEagerTensorFromArgs("dgc_momentum", "Grad", args, 3, false);
    auto Velocity =
        GetEagerTensorFromArgs("dgc_momentum", "Velocity", args, 4, false);
    auto LearningRate =
        GetEagerTensorFromArgs("dgc_momentum", "LearningRate", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("dgc_momentum", args, 6, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = dgc_momentum_dygraph_function(current_step, nranks, Param, Grad,
                                             Velocity, LearningRate, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lamb(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("lamb", "Param", args, 0, false);
    auto Grad = GetEagerTensorFromArgs("lamb", "Grad", args, 1, false);
    auto LearningRate =
        GetEagerTensorFromArgs("lamb", "LearningRate", args, 2, false);
    auto Moment1 = GetEagerTensorFromArgs("lamb", "Moment1", args, 3, false);
    auto Moment2 = GetEagerTensorFromArgs("lamb", "Moment2", args, 4, false);
    auto Beta1Pow = GetEagerTensorFromArgs("lamb", "Beta1Pow", args, 5, false);
    auto Beta2Pow = GetEagerTensorFromArgs("lamb", "Beta2Pow", args, 6, false);
    auto ParamOut = GetEagerTensorFromArgs("lamb", "ParamOut", args, 7, false);
    auto Moment1Out =
        GetEagerTensorFromArgs("lamb", "Moment1Out", args, 8, false);
    auto Moment2Out =
        GetEagerTensorFromArgs("lamb", "Moment2Out", args, 9, false);
    auto Beta1PowOut =
        GetEagerTensorFromArgs("lamb", "Beta1PowOut", args, 10, true);
    auto Beta2PowOut =
        GetEagerTensorFromArgs("lamb", "Beta2PowOut", args, 11, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lamb", args, 12, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = lamb_dygraph_function(Param, Grad, LearningRate, Moment1,
                                     Moment2, Beta1Pow, Beta2Pow, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_generate_proposals_v2(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Scores = GetEagerTensorFromArgs("generate_proposals_v2", "Scores",
                                         args, 0, false);
    auto BboxDeltas = GetEagerTensorFromArgs("generate_proposals_v2",
                                             "BboxDeltas", args, 1, false);
    auto ImShape = GetEagerTensorFromArgs("generate_proposals_v2", "ImShape",
                                          args, 2, false);
    auto Anchors = GetEagerTensorFromArgs("generate_proposals_v2", "Anchors",
                                          args, 3, false);
    auto Variances = GetEagerTensorFromArgs("generate_proposals_v2",
                                            "Variances", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("generate_proposals_v2", args, 5,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = generate_proposals_v2_dygraph_function(
        Scores, BboxDeltas, ImShape, Anchors, Variances, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bitwise_or(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bitwise_or", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("bitwise_or", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bitwise_or", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = bitwise_or_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_gru_unit(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("gru_unit", "Input", args, 0, false);
    auto HiddenPrev =
        GetEagerTensorFromArgs("gru_unit", "HiddenPrev", args, 1, false);
    auto Weight = GetEagerTensorFromArgs("gru_unit", "Weight", args, 2, false);
    auto Bias = GetEagerTensorFromArgs("gru_unit", "Bias", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("gru_unit", args, 4, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = gru_unit_dygraph_function(Input, HiddenPrev, Weight, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fake_channel_wise_quantize_dequantize_abs_max(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs(
        "fake_channel_wise_quantize_dequantize_abs_max", "X", args, 0, false);
    auto Out = GetEagerTensorFromArgs(
        "fake_channel_wise_quantize_dequantize_abs_max", "Out", args, 1, false);
    auto OutScale =
        GetEagerTensorFromArgs("fake_channel_wise_quantize_dequantize_abs_max",
                               "OutScale", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fake_channel_wise_quantize_dequantize_abs_max",
                               args, 3, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fake_channel_wise_quantize_dequantize_abs_max_dygraph_function(
        X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sampling_id(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sampling_id", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sampling_id", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = sampling_id_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_unsqueeze2(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("unsqueeze2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("unsqueeze2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = unsqueeze2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_average_accumulates(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto param =
        GetEagerTensorFromArgs("average_accumulates", "param", args, 0, false);
    auto in_sum_1 = GetEagerTensorFromArgs("average_accumulates", "in_sum_1",
                                           args, 1, false);
    auto in_sum_2 = GetEagerTensorFromArgs("average_accumulates", "in_sum_2",
                                           args, 2, false);
    auto in_sum_3 = GetEagerTensorFromArgs("average_accumulates", "in_sum_3",
                                           args, 3, false);
    auto in_num_accumulates = GetEagerTensorFromArgs(
        "average_accumulates", "in_num_accumulates", args, 4, false);
    auto in_old_num_accumulates = GetEagerTensorFromArgs(
        "average_accumulates", "in_old_num_accumulates", args, 5, false);
    auto in_num_updates = GetEagerTensorFromArgs(
        "average_accumulates", "in_num_updates", args, 6, false);
    auto out_sum_1 = GetEagerTensorFromArgs("average_accumulates", "out_sum_1",
                                            args, 7, false);
    auto out_sum_2 = GetEagerTensorFromArgs("average_accumulates", "out_sum_2",
                                            args, 8, false);
    auto out_sum_3 = GetEagerTensorFromArgs("average_accumulates", "out_sum_3",
                                            args, 9, false);
    auto out_num_accumulates = GetEagerTensorFromArgs(
        "average_accumulates", "out_num_accumulates", args, 10, false);
    auto out_old_num_accumulates = GetEagerTensorFromArgs(
        "average_accumulates", "out_old_num_accumulates", args, 11, false);
    auto out_num_updates = GetEagerTensorFromArgs(
        "average_accumulates", "out_num_updates", args, 12, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("average_accumulates", args, 13,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = average_accumulates_dygraph_function(
        param, in_sum_1, in_sum_2, in_sum_3, in_num_accumulates,
        in_old_num_accumulates, in_num_updates, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_enumerate(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_enumerate", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_enumerate", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_enumerate_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fusion_seqconv_eltadd_relu(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fusion_seqconv_eltadd_relu", "X", args, 0,
                                    false);
    auto Filter = GetEagerTensorFromArgs("fusion_seqconv_eltadd_relu", "Filter",
                                         args, 1, false);
    auto Bias = GetEagerTensorFromArgs("fusion_seqconv_eltadd_relu", "Bias",
                                       args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fusion_seqconv_eltadd_relu", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        fusion_seqconv_eltadd_relu_dygraph_function(X, Filter, Bias, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bce_loss(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bce_loss", "X", args, 0, false);
    auto Label = GetEagerTensorFromArgs("bce_loss", "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bce_loss", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = bce_loss_dygraph_function(X, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_generate_proposal_labels(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto RpnRois = GetEagerTensorFromArgs("generate_proposal_labels", "RpnRois",
                                          args, 0, false);
    auto GtClasses = GetEagerTensorFromArgs("generate_proposal_labels",
                                            "GtClasses", args, 1, false);
    auto IsCrowd = GetEagerTensorFromArgs("generate_proposal_labels", "IsCrowd",
                                          args, 2, false);
    auto GtBoxes = GetEagerTensorFromArgs("generate_proposal_labels", "GtBoxes",
                                          args, 3, false);
    auto ImInfo = GetEagerTensorFromArgs("generate_proposal_labels", "ImInfo",
                                         args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("generate_proposal_labels", args, 5,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = generate_proposal_labels_dygraph_function(
        RpnRois, GtClasses, IsCrowd, GtBoxes, ImInfo, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_im2sequence(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("im2sequence", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("im2sequence", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = im2sequence_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_isinf(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("isinf", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("isinf", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = isinf_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_adagrad(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("adagrad", "Param", args, 0, false);
    auto Grad = GetEagerTensorFromArgs("adagrad", "Grad", args, 1, false);
    auto Moment = GetEagerTensorFromArgs("adagrad", "Moment", args, 2, false);
    auto LearningRate =
        GetEagerTensorFromArgs("adagrad", "LearningRate", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("adagrad", args, 4, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out =
        adagrad_dygraph_function(Param, Grad, Moment, LearningRate, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_linear_chain_crf(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Emission =
        GetEagerTensorFromArgs("linear_chain_crf", "Emission", args, 0, false);
    auto Transition = GetEagerTensorFromArgs("linear_chain_crf", "Transition",
                                             args, 1, false);
    auto Label =
        GetEagerTensorFromArgs("linear_chain_crf", "Label", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("linear_chain_crf", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        linear_chain_crf_dygraph_function(Emission, Transition, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_retinanet_target_assign(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Anchor = GetEagerTensorFromArgs("retinanet_target_assign", "Anchor",
                                         args, 0, false);
    auto GtBoxes = GetEagerTensorFromArgs("retinanet_target_assign", "GtBoxes",
                                          args, 1, false);
    auto GtLabels = GetEagerTensorFromArgs("retinanet_target_assign",
                                           "GtLabels", args, 2, false);
    auto IsCrowd = GetEagerTensorFromArgs("retinanet_target_assign", "IsCrowd",
                                          args, 3, false);
    auto ImInfo = GetEagerTensorFromArgs("retinanet_target_assign", "ImInfo",
                                         args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("retinanet_target_assign", args, 5,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = retinanet_target_assign_dygraph_function(
        Anchor, GtBoxes, GtLabels, IsCrowd, ImInfo, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fusion_group(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Inputs =
        GetEagerTensorListFromArgs("fusion_group", "Inputs", args, 0, false);
    auto OutsNum =
        GetUnsignedLongFromArgs("fusion_group", "OutsNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fusion_group", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fusion_group_dygraph_function(Inputs, OutsNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_teacher_student_sigmoid_loss(PyObject *self,
                                                        PyObject *args,
                                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("teacher_student_sigmoid_loss", "X", args,
                                    0, false);
    auto Label = GetEagerTensorFromArgs("teacher_student_sigmoid_loss", "Label",
                                        args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("teacher_student_sigmoid_loss", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = teacher_student_sigmoid_loss_dygraph_function(X, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_random_crop(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("random_crop", "X", args, 0, false);
    auto Seed = GetEagerTensorFromArgs("random_crop", "Seed", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("random_crop", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = random_crop_dygraph_function(X, Seed, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lookup_table_v2(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto W = GetEagerTensorFromArgs("lookup_table_v2", "W", args, 0, false);
    auto Ids = GetEagerTensorFromArgs("lookup_table_v2", "Ids", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lookup_table_v2", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = lookup_table_v2_dygraph_function(W, Ids, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_detection_map(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto DetectRes =
        GetEagerTensorFromArgs("detection_map", "DetectRes", args, 0, false);
    auto Label =
        GetEagerTensorFromArgs("detection_map", "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("detection_map", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = detection_map_dygraph_function(DetectRes, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_l1_norm(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("l1_norm", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("l1_norm", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = l1_norm_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sqrt(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sqrt", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sqrt", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sqrt_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_elemwise_activation(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fused_elemwise_activation", "X", args, 0,
                                    false);
    auto Y = GetEagerTensorFromArgs("fused_elemwise_activation", "Y", args, 1,
                                    false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_elemwise_activation", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_elemwise_activation_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_slogdeterminant(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("slogdeterminant", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("slogdeterminant", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = slogdeterminant_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_share_buffer(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("share_buffer", "X", args, 0, false);
    auto OutNum =
        GetUnsignedLongFromArgs("share_buffer", "OutNum", args, 1, false);
    auto XOutNum =
        GetUnsignedLongFromArgs("share_buffer", "XOutNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("share_buffer", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = share_buffer_dygraph_function(X, OutNum, XOutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bitwise_and(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bitwise_and", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("bitwise_and", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bitwise_and", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = bitwise_and_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_diag_embed(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("diag_embed", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("diag_embed", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = diag_embed_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_unbind(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("unbind", "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs("unbind", "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("unbind", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = unbind_dygraph_function(X, OutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_dropout(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("dropout", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("dropout", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = dropout_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_moving_average_abs_max_scale(PyObject *self,
                                                        PyObject *args,
                                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("moving_average_abs_max_scale", "X", args,
                                    0, false);
    auto InAccum = GetEagerTensorFromArgs("moving_average_abs_max_scale",
                                          "InAccum", args, 1, true);
    auto InState = GetEagerTensorFromArgs("moving_average_abs_max_scale",
                                          "InState", args, 2, true);
    auto Out = GetEagerTensorFromArgs("moving_average_abs_max_scale", "Out",
                                      args, 3, true);
    auto OutScale = GetEagerTensorFromArgs("moving_average_abs_max_scale",
                                           "OutScale", args, 4, false);
    auto OutState = GetEagerTensorFromArgs("moving_average_abs_max_scale",
                                           "OutState", args, 5, true);
    auto OutAccum = GetEagerTensorFromArgs("moving_average_abs_max_scale",
                                           "OutAccum", args, 6, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("moving_average_abs_max_scale", args, 7,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = moving_average_abs_max_scale_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_beam_search(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto pre_ids =
        GetEagerTensorFromArgs("beam_search", "pre_ids", args, 0, false);
    auto pre_scores =
        GetEagerTensorFromArgs("beam_search", "pre_scores", args, 1, false);
    auto scores =
        GetEagerTensorFromArgs("beam_search", "scores", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("beam_search", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = beam_search_dygraph_function(pre_ids, pre_scores, scores, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_log_loss(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Predicted =
        GetEagerTensorFromArgs("log_loss", "Predicted", args, 0, false);
    auto Labels = GetEagerTensorFromArgs("log_loss", "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("log_loss", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = log_loss_dygraph_function(Predicted, Labels, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_greater_than(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("greater_than", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("greater_than", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("greater_than", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = greater_than_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_kron(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("kron", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("kron", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("kron", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = kron_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sigmoid_focal_loss(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sigmoid_focal_loss", "X", args, 0, false);
    auto Label =
        GetEagerTensorFromArgs("sigmoid_focal_loss", "Label", args, 1, false);
    auto FgNum =
        GetEagerTensorFromArgs("sigmoid_focal_loss", "FgNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sigmoid_focal_loss", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sigmoid_focal_loss_dygraph_function(X, Label, FgNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_rmsprop(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("rmsprop", "Param", args, 0, false);
    auto MeanSquare =
        GetEagerTensorFromArgs("rmsprop", "MeanSquare", args, 1, false);
    auto LearningRate =
        GetEagerTensorFromArgs("rmsprop", "LearningRate", args, 2, false);
    auto Grad = GetEagerTensorFromArgs("rmsprop", "Grad", args, 3, false);
    auto Moment = GetEagerTensorFromArgs("rmsprop", "Moment", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("rmsprop", args, 5, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = rmsprop_dygraph_function(Param, MeanSquare, LearningRate, Grad,
                                        Moment, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_conv2d(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("conv2d", "Input", args, 0, false);
    auto Filter = GetEagerTensorFromArgs("conv2d", "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("conv2d", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = conv2d_dygraph_function(Input, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_uniform_random_inplace(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("uniform_random_inplace", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("uniform_random_inplace", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = uniform_random_inplace_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_maxout(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("maxout", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("maxout", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = maxout_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_linear_interp(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("linear_interp", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("linear_interp", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = linear_interp_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_auc(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Predict = GetEagerTensorFromArgs("auc", "Predict", args, 0, false);
    auto Label = GetEagerTensorFromArgs("auc", "Label", args, 1, false);
    auto StatPos = GetEagerTensorFromArgs("auc", "StatPos", args, 2, false);
    auto StatNeg = GetEagerTensorFromArgs("auc", "StatNeg", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("auc", args, 4, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = auc_dygraph_function(Predict, Label, StatPos, StatNeg, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_logical_or(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("logical_or", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("logical_or", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("logical_or", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = logical_or_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_batch_norm(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("batch_norm", "X", args, 0, false);
    auto Scale = GetEagerTensorFromArgs("batch_norm", "Scale", args, 1, false);
    auto Bias = GetEagerTensorFromArgs("batch_norm", "Bias", args, 2, false);
    auto Mean = GetEagerTensorFromArgs("batch_norm", "Mean", args, 3, false);
    auto Variance =
        GetEagerTensorFromArgs("batch_norm", "Variance", args, 4, false);
    auto MeanOut =
        GetEagerTensorFromArgs("batch_norm", "MeanOut", args, 5, false);
    auto VarianceOut =
        GetEagerTensorFromArgs("batch_norm", "VarianceOut", args, 6, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("batch_norm", args, 7, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out =
        batch_norm_dygraph_function(X, Scale, Bias, Mean, Variance, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elementwise_add(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("elementwise_add", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("elementwise_add", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elementwise_add", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elementwise_add_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_acos(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("acos", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("acos", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = acos_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_unpool(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("unpool", "X", args, 0, false);
    auto Indices = GetEagerTensorFromArgs("unpool", "Indices", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("unpool", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = unpool_dygraph_function(X, Indices, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cumprod(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cumprod", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cumprod", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = cumprod_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sample_logits(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Logits =
        GetEagerTensorFromArgs("sample_logits", "Logits", args, 0, false);
    auto Labels =
        GetEagerTensorFromArgs("sample_logits", "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sample_logits", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = sample_logits_dygraph_function(Logits, Labels, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pull_box_extended_sparse(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Ids = GetEagerTensorListFromArgs("pull_box_extended_sparse", "Ids",
                                          args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs("pull_box_extended_sparse", "OutNum",
                                          args, 1, false);
    auto OutExtendNum = GetUnsignedLongFromArgs("pull_box_extended_sparse",
                                                "OutExtendNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pull_box_extended_sparse", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = pull_box_extended_sparse_dygraph_function(Ids, OutNum,
                                                         OutExtendNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_crop_tensor(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("crop_tensor", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("crop_tensor", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = crop_tensor_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fill_constant(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Out = GetEagerTensorFromArgs("fill_constant", "Out", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fill_constant", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fill_constant_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_deformable_conv(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("deformable_conv", "Input", args, 0, false);
    auto Offset =
        GetEagerTensorFromArgs("deformable_conv", "Offset", args, 1, false);
    auto Mask =
        GetEagerTensorFromArgs("deformable_conv", "Mask", args, 2, false);
    auto Filter =
        GetEagerTensorFromArgs("deformable_conv", "Filter", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("deformable_conv", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        deformable_conv_dygraph_function(Input, Offset, Mask, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_generate_mask_labels(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto ImInfo = GetEagerTensorFromArgs("generate_mask_labels", "ImInfo", args,
                                         0, false);
    auto GtClasses = GetEagerTensorFromArgs("generate_mask_labels", "GtClasses",
                                            args, 1, false);
    auto IsCrowd = GetEagerTensorFromArgs("generate_mask_labels", "IsCrowd",
                                          args, 2, false);
    auto GtSegms = GetEagerTensorFromArgs("generate_mask_labels", "GtSegms",
                                          args, 3, false);
    auto Rois =
        GetEagerTensorFromArgs("generate_mask_labels", "Rois", args, 4, false);
    auto LabelsInt32 = GetEagerTensorFromArgs("generate_mask_labels",
                                              "LabelsInt32", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("generate_mask_labels", args, 6,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = generate_mask_labels_dygraph_function(
        ImInfo, GtClasses, IsCrowd, GtSegms, Rois, LabelsInt32, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_locality_aware_nms(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto BBoxes =
        GetEagerTensorFromArgs("locality_aware_nms", "BBoxes", args, 0, false);
    auto Scores =
        GetEagerTensorFromArgs("locality_aware_nms", "Scores", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("locality_aware_nms", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = locality_aware_nms_dygraph_function(BBoxes, Scores, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_expand_as(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("expand_as", "X", args, 0, false);
    auto target_tensor =
        GetEagerTensorFromArgs("expand_as", "target_tensor", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("expand_as", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = expand_as_dygraph_function(X, target_tensor, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_matrix_power(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("matrix_power", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("matrix_power", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = matrix_power_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_greater_equal(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("greater_equal", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("greater_equal", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("greater_equal", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = greater_equal_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_generate_proposals(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Scores =
        GetEagerTensorFromArgs("generate_proposals", "Scores", args, 0, false);
    auto BboxDeltas = GetEagerTensorFromArgs("generate_proposals", "BboxDeltas",
                                             args, 1, false);
    auto ImInfo =
        GetEagerTensorFromArgs("generate_proposals", "ImInfo", args, 2, false);
    auto Anchors =
        GetEagerTensorFromArgs("generate_proposals", "Anchors", args, 3, false);
    auto Variances = GetEagerTensorFromArgs("generate_proposals", "Variances",
                                            args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("generate_proposals", args, 5,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = generate_proposals_dygraph_function(Scores, BboxDeltas, ImInfo,
                                                   Anchors, Variances, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bilinear_interp(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bilinear_interp", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bilinear_interp", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = bilinear_interp_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sigmoid(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sigmoid", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sigmoid", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = sigmoid_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_inplace_abn(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("inplace_abn", "X", args, 0, false);
    auto Scale = GetEagerTensorFromArgs("inplace_abn", "Scale", args, 1, false);
    auto Bias = GetEagerTensorFromArgs("inplace_abn", "Bias", args, 2, false);
    auto Mean = GetEagerTensorFromArgs("inplace_abn", "Mean", args, 3, false);
    auto Variance =
        GetEagerTensorFromArgs("inplace_abn", "Variance", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("inplace_abn", args, 5, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out =
        inplace_abn_dygraph_function(X, Scale, Bias, Mean, Variance, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_softshrink(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("softshrink", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("softshrink", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = softshrink_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_mul(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("mul", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("mul", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("mul", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = mul_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_data_norm(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("data_norm", "X", args, 0, false);
    auto BatchSize =
        GetEagerTensorFromArgs("data_norm", "BatchSize", args, 1, false);
    auto BatchSum =
        GetEagerTensorFromArgs("data_norm", "BatchSum", args, 2, false);
    auto BatchSquareSum =
        GetEagerTensorFromArgs("data_norm", "BatchSquareSum", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("data_norm", args, 4, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = data_norm_dygraph_function(X, BatchSize, BatchSum,
                                          BatchSquareSum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_get_tensor_from_selected_rows(PyObject *self,
                                                         PyObject *args,
                                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("get_tensor_from_selected_rows", "X", args,
                                    0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("get_tensor_from_selected_rows", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = get_tensor_from_selected_rows_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_spp(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("spp", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("spp", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = spp_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_floor(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("floor", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("floor", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = floor_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_gelu(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("gelu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("gelu", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = gelu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_retinanet_detection_output(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto BBoxes = GetEagerTensorListFromArgs("retinanet_detection_output",
                                             "BBoxes", args, 0, false);
    auto Scores = GetEagerTensorListFromArgs("retinanet_detection_output",
                                             "Scores", args, 1, false);
    auto Anchors = GetEagerTensorListFromArgs("retinanet_detection_output",
                                              "Anchors", args, 2, false);
    auto ImInfo = GetEagerTensorFromArgs("retinanet_detection_output", "ImInfo",
                                         args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("retinanet_detection_output", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = retinanet_detection_output_dygraph_function(
        BBoxes, Scores, Anchors, ImInfo, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_minus(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("minus", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("minus", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("minus", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = minus_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_push_dense(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Ids = GetEagerTensorListFromArgs("push_dense", "Ids", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("push_dense", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = push_dense_dygraph_function(Ids, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    Py_INCREF(Py_None);
    return Py_None;
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_silu(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("silu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("silu", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = silu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_erase(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_erase", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_erase", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_erase_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_real(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("real", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("real", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = real_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_nearest_interp_v2(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("nearest_interp_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("nearest_interp_v2", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = nearest_interp_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_dgc_clip_by_norm(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto current_step = GetEagerTensorFromArgs("dgc_clip_by_norm",
                                               "current_step", args, 0, false);
    auto X = GetEagerTensorFromArgs("dgc_clip_by_norm", "X", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("dgc_clip_by_norm", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = dgc_clip_by_norm_dygraph_function(current_step, X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_squeeze2(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("squeeze2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("squeeze2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = squeeze2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_strided_slice(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("strided_slice", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("strided_slice", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = strided_slice_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_conj(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("conj", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("conj", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = conj_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_precision_recall(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto MaxProbs =
        GetEagerTensorFromArgs("precision_recall", "MaxProbs", args, 0, false);
    auto Indices =
        GetEagerTensorFromArgs("precision_recall", "Indices", args, 1, false);
    auto Labels =
        GetEagerTensorFromArgs("precision_recall", "Labels", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("precision_recall", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        precision_recall_dygraph_function(MaxProbs, Indices, Labels, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_save(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("save", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("save", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = save_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    Py_INCREF(Py_None);
    return Py_None;
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fusion_seqexpand_concat_fc(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("fusion_seqexpand_concat_fc", "X", args,
                                        0, false);
    auto FCWeight = GetEagerTensorFromArgs("fusion_seqexpand_concat_fc",
                                           "FCWeight", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fusion_seqexpand_concat_fc", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fusion_seqexpand_concat_fc_dygraph_function(X, FCWeight, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fake_quantize_range_abs_max(PyObject *self,
                                                       PyObject *args,
                                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fake_quantize_range_abs_max", "X", args, 0,
                                    false);
    auto InScale = GetEagerTensorFromArgs("fake_quantize_range_abs_max",
                                          "InScale", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fake_quantize_range_abs_max", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fake_quantize_range_abs_max_dygraph_function(X, InScale, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_depthwise_conv2d_transpose(PyObject *self,
                                                      PyObject *args,
                                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("depthwise_conv2d_transpose", "Input",
                                        args, 0, false);
    auto Filter = GetEagerTensorFromArgs("depthwise_conv2d_transpose", "Filter",
                                         args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("depthwise_conv2d_transpose", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        depthwise_conv2d_transpose_dygraph_function(Input, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_positive_negative_pair(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Score = GetEagerTensorFromArgs("positive_negative_pair", "Score", args,
                                        0, false);
    auto Label = GetEagerTensorFromArgs("positive_negative_pair", "Label", args,
                                        1, false);
    auto QueryID = GetEagerTensorFromArgs("positive_negative_pair", "QueryID",
                                          args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("positive_negative_pair", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        positive_negative_pair_dygraph_function(Score, Label, QueryID, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_square(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("square", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("square", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = square_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_var_conv_2d(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("var_conv_2d", "X", args, 0, false);
    auto ROW = GetEagerTensorFromArgs("var_conv_2d", "ROW", args, 1, false);
    auto COLUMN =
        GetEagerTensorFromArgs("var_conv_2d", "COLUMN", args, 2, false);
    auto W = GetEagerTensorFromArgs("var_conv_2d", "W", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("var_conv_2d", args, 4, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = var_conv_2d_dygraph_function(X, ROW, COLUMN, W, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_log1p(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("log1p", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("log1p", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = log1p_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_softmax_mask_upper_triangle(PyObject *self,
                                                             PyObject *args,
                                                             PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fused_softmax_mask_upper_triangle", "X",
                                    args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_softmax_mask_upper_triangle", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_softmax_mask_upper_triangle_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_clip_by_norm(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("clip_by_norm", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("clip_by_norm", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = clip_by_norm_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_atan2(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X1 = GetEagerTensorFromArgs("atan2", "X1", args, 0, false);
    auto X2 = GetEagerTensorFromArgs("atan2", "X2", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("atan2", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = atan2_dygraph_function(X1, X2, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_box_decoder_and_assign(PyObject *self,
                                                  PyObject *args,
                                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto PriorBox = GetEagerTensorFromArgs("box_decoder_and_assign", "PriorBox",
                                           args, 0, false);
    auto TargetBox = GetEagerTensorFromArgs("box_decoder_and_assign",
                                            "TargetBox", args, 1, false);
    auto BoxScore = GetEagerTensorFromArgs("box_decoder_and_assign", "BoxScore",
                                           args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("box_decoder_and_assign", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = box_decoder_and_assign_dygraph_function(PriorBox, TargetBox,
                                                       BoxScore, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fft_r2c(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fft_r2c", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fft_r2c", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fft_r2c_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_roi_pool(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("roi_pool", "X", args, 0, false);
    auto ROIs = GetEagerTensorFromArgs("roi_pool", "ROIs", args, 1, false);
    auto RoisNum = GetEagerTensorFromArgs("roi_pool", "RoisNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("roi_pool", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = roi_pool_dygraph_function(X, ROIs, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_overlap_add(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("overlap_add", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("overlap_add", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = overlap_add_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fill_constant_batch_size_like(PyObject *self,
                                                         PyObject *args,
                                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("fill_constant_batch_size_like",
                                        "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fill_constant_batch_size_like", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fill_constant_batch_size_like_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fill_any(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fill_any", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fill_any", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fill_any_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_dequantize_log(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("dequantize_log", "X", args, 0, false);
    auto Dict =
        GetEagerTensorFromArgs("dequantize_log", "Dict", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("dequantize_log", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = dequantize_log_dygraph_function(X, Dict, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_max_pool2d_with_index(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("max_pool2d_with_index", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("max_pool2d_with_index", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = max_pool2d_with_index_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pad3d(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("pad3d", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pad3d", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = pad3d_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_norm(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("norm", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("norm", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = norm_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_viterbi_decode(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("viterbi_decode", "Input", args, 0, false);
    auto Transition =
        GetEagerTensorFromArgs("viterbi_decode", "Transition", args, 1, false);
    auto Length =
        GetEagerTensorFromArgs("viterbi_decode", "Length", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("viterbi_decode", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        viterbi_decode_dygraph_function(Input, Transition, Length, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_mish(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("mish", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("mish", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = mish_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_box_coder(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto PriorBox =
        GetEagerTensorFromArgs("box_coder", "PriorBox", args, 0, false);
    auto PriorBoxVar =
        GetEagerTensorFromArgs("box_coder", "PriorBoxVar", args, 1, true);
    auto TargetBox =
        GetEagerTensorFromArgs("box_coder", "TargetBox", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("box_coder", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = box_coder_dygraph_function(PriorBox, TargetBox, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_flatten(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("flatten", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("flatten", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = flatten_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elementwise_mod(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("elementwise_mod", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("elementwise_mod", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elementwise_mod", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elementwise_mod_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_margin_cross_entropy(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Logits = GetEagerTensorFromArgs("margin_cross_entropy", "Logits", args,
                                         0, false);
    auto Label =
        GetEagerTensorFromArgs("margin_cross_entropy", "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("margin_cross_entropy", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = margin_cross_entropy_dygraph_function(Logits, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pull_sparse(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Ids = GetEagerTensorListFromArgs("pull_sparse", "Ids", args, 0, false);
    auto W = GetEagerTensorListFromArgs("pull_sparse", "W", args, 1, false);
    auto OutNum =
        GetUnsignedLongFromArgs("pull_sparse", "OutNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pull_sparse", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = pull_sparse_dygraph_function(Ids, W, OutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_logical_and(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("logical_and", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("logical_and", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("logical_and", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = logical_and_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pow(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("pow", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pow", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = pow_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_stanh(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("stanh", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("stanh", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = stanh_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_label_smooth(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("label_smooth", "X", args, 0, false);
    auto PriorDist =
        GetEagerTensorFromArgs("label_smooth", "PriorDist", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("label_smooth", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = label_smooth_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_merged_momentum(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param =
        GetEagerTensorListFromArgs("merged_momentum", "Param", args, 0, false);
    auto Grad =
        GetEagerTensorListFromArgs("merged_momentum", "Grad", args, 1, false);
    auto Velocity = GetEagerTensorListFromArgs("merged_momentum", "Velocity",
                                               args, 2, false);
    auto LearningRate = GetEagerTensorListFromArgs(
        "merged_momentum", "LearningRate", args, 3, false);
    auto ParamOutNum = GetUnsignedLongFromArgs("merged_momentum", "ParamOutNum",
                                               args, 4, false);
    auto VelocityOutNum = GetUnsignedLongFromArgs(
        "merged_momentum", "VelocityOutNum", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("merged_momentum", args, 6,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        merged_momentum_dygraph_function(Param, Grad, Velocity, LearningRate,
                                         ParamOutNum, VelocityOutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_ascend_trigger(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto FeedList = GetEagerTensorListFromArgs("ascend_trigger", "FeedList",
                                               args, 0, false);
    auto FetchListNum = GetUnsignedLongFromArgs("ascend_trigger",
                                                "FetchListNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("ascend_trigger", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = ascend_trigger_dygraph_function(FeedList, FetchListNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_feedforward(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fused_feedforward", "X", args, 0, false);
    auto Linear1Weight = GetEagerTensorFromArgs(
        "fused_feedforward", "Linear1Weight", args, 1, false);
    auto Linear2Weight = GetEagerTensorFromArgs(
        "fused_feedforward", "Linear2Weight", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_feedforward", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_feedforward_dygraph_function(X, Linear1Weight,
                                                  Linear2Weight, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_rpn_target_assign(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Anchor =
        GetEagerTensorFromArgs("rpn_target_assign", "Anchor", args, 0, false);
    auto GtBoxes =
        GetEagerTensorFromArgs("rpn_target_assign", "GtBoxes", args, 1, false);
    auto IsCrowd =
        GetEagerTensorFromArgs("rpn_target_assign", "IsCrowd", args, 2, false);
    auto ImInfo =
        GetEagerTensorFromArgs("rpn_target_assign", "ImInfo", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("rpn_target_assign", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = rpn_target_assign_dygraph_function(Anchor, GtBoxes, IsCrowd,
                                                  ImInfo, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_roi_perspective_transform(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("roi_perspective_transform", "X", args, 0,
                                    false);
    auto ROIs = GetEagerTensorFromArgs("roi_perspective_transform", "ROIs",
                                       args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("roi_perspective_transform", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = roi_perspective_transform_dygraph_function(X, ROIs, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_expand(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("expand", "X", args, 0, false);
    auto ExpandTimes =
        GetEagerTensorFromArgs("expand", "ExpandTimes", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("expand", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = expand_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_prroi_pool(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("prroi_pool", "X", args, 0, false);
    auto ROIs = GetEagerTensorFromArgs("prroi_pool", "ROIs", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("prroi_pool", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = prroi_pool_dygraph_function(X, ROIs, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pool3d(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("pool3d", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pool3d", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = pool3d_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_memcpy(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("memcpy", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("memcpy", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = memcpy_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_distribute_fpn_proposals(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto FpnRois = GetEagerTensorFromArgs("distribute_fpn_proposals", "FpnRois",
                                          args, 0, false);
    auto RoisNum = GetEagerTensorFromArgs("distribute_fpn_proposals", "RoisNum",
                                          args, 1, true);
    auto MultiFpnRoisNum = GetUnsignedLongFromArgs(
        "distribute_fpn_proposals", "MultiFpnRoisNum", args, 2, false);
    auto MultiLevelRoIsNumNum = GetUnsignedLongFromArgs(
        "distribute_fpn_proposals", "MultiLevelRoIsNumNum", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("distribute_fpn_proposals", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = distribute_fpn_proposals_dygraph_function(
        FpnRois, MultiFpnRoisNum, MultiLevelRoIsNumNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_frame(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("frame", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("frame", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = frame_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bincount(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bincount", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bincount", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = bincount_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_shape(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("shape", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("shape", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = shape_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_group_norm(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("group_norm", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("group_norm", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = group_norm_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_resnet_unit(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("resnet_unit", "X", args, 0, false);
    auto FilterX =
        GetEagerTensorFromArgs("resnet_unit", "FilterX", args, 1, false);
    auto ScaleX =
        GetEagerTensorFromArgs("resnet_unit", "ScaleX", args, 2, false);
    auto BiasX = GetEagerTensorFromArgs("resnet_unit", "BiasX", args, 3, false);
    auto MeanX = GetEagerTensorFromArgs("resnet_unit", "MeanX", args, 4, false);
    auto VarX = GetEagerTensorFromArgs("resnet_unit", "VarX", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("resnet_unit", args, 6, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = resnet_unit_dygraph_function(X, FilterX, ScaleX, BiasX, MeanX,
                                            VarX, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_expand_as(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_expand_as", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("sequence_expand_as", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_expand_as", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_expand_as_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cos_sim(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cos_sim", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("cos_sim", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cos_sim", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = cos_sim_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_eigvals(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("eigvals", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("eigvals", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = eigvals_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_save_combine(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("save_combine", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("save_combine", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = save_combine_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    Py_INCREF(Py_None);
    return Py_None;
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_class_center_sample(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Label =
        GetEagerTensorFromArgs("class_center_sample", "Label", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("class_center_sample", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = class_center_sample_dygraph_function(Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_read_file(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("read_file", args, 0, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = read_file_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_isfinite(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("isfinite", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("isfinite", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = isfinite_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_arg_max(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("arg_max", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("arg_max", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = arg_max_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_equal(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("equal", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("equal", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("equal", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = equal_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fake_dequantize_max_abs(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("fake_dequantize_max_abs", "X", args, 0, false);
    auto Scale = GetEagerTensorFromArgs("fake_dequantize_max_abs", "Scale",
                                        args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fake_dequantize_max_abs", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fake_dequantize_max_abs_dygraph_function(X, Scale, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_qr(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("qr", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("qr", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = qr_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_anchor_generator(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("anchor_generator", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("anchor_generator", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = anchor_generator_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_layer_norm(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("layer_norm", "X", args, 0, false);
    auto Scale = GetEagerTensorFromArgs("layer_norm", "Scale", args, 1, true);
    auto Bias = GetEagerTensorFromArgs("layer_norm", "Bias", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("layer_norm", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = layer_norm_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_merge_selected_rows(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("merge_selected_rows", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("merge_selected_rows", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = merge_selected_rows_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_less_equal(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("less_equal", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("less_equal", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("less_equal", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = less_equal_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_rnn(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("rnn", "Input", args, 0, false);
    auto PreState =
        GetEagerTensorListFromArgs("rnn", "PreState", args, 1, false);
    auto WeightList =
        GetEagerTensorListFromArgs("rnn", "WeightList", args, 2, false);
    auto SequenceLength =
        GetEagerTensorFromArgs("rnn", "SequenceLength", args, 3, true);
    auto DropoutState =
        GetEagerTensorFromArgs("rnn", "DropoutState", args, 4, true);
    auto StateNum = GetUnsignedLongFromArgs("rnn", "StateNum", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("rnn", args, 6, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        rnn_dygraph_function(Input, PreState, WeightList, StateNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fusion_lstm(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fusion_lstm", "X", args, 0, false);
    auto WeightX =
        GetEagerTensorFromArgs("fusion_lstm", "WeightX", args, 1, false);
    auto WeightH =
        GetEagerTensorFromArgs("fusion_lstm", "WeightH", args, 2, false);
    auto Bias = GetEagerTensorFromArgs("fusion_lstm", "Bias", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fusion_lstm", args, 4, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fusion_lstm_dygraph_function(X, WeightX, WeightH, Bias, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lars_momentum(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param =
        GetEagerTensorListFromArgs("lars_momentum", "Param", args, 0, false);
    auto Grad =
        GetEagerTensorListFromArgs("lars_momentum", "Grad", args, 1, false);
    auto Velocity =
        GetEagerTensorListFromArgs("lars_momentum", "Velocity", args, 2, false);
    auto LearningRate = GetEagerTensorListFromArgs(
        "lars_momentum", "LearningRate", args, 3, false);
    auto ParamOutNum =
        GetUnsignedLongFromArgs("lars_momentum", "ParamOutNum", args, 4, false);
    auto VelocityOutNum = GetUnsignedLongFromArgs(
        "lars_momentum", "VelocityOutNum", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lars_momentum", args, 6, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out =
        lars_momentum_dygraph_function(Param, Grad, Velocity, LearningRate,
                                       ParamOutNum, VelocityOutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_hard_sigmoid(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("hard_sigmoid", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("hard_sigmoid", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = hard_sigmoid_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_isnan(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("isnan", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("isnan", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = isnan_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elementwise_floordiv(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("elementwise_floordiv", "X", args, 0, false);
    auto Y =
        GetEagerTensorFromArgs("elementwise_floordiv", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elementwise_floordiv", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elementwise_floordiv_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_correlation(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input1 =
        GetEagerTensorFromArgs("correlation", "Input1", args, 0, false);
    auto Input2 =
        GetEagerTensorFromArgs("correlation", "Input2", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("correlation", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = correlation_dygraph_function(Input1, Input2, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_histogram(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("histogram", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("histogram", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = histogram_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_gather_tree(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Ids = GetEagerTensorFromArgs("gather_tree", "Ids", args, 0, false);
    auto Parents =
        GetEagerTensorFromArgs("gather_tree", "Parents", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("gather_tree", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = gather_tree_dygraph_function(Ids, Parents, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_segment_pool(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("segment_pool", "X", args, 0, false);
    auto SegmentIds =
        GetEagerTensorFromArgs("segment_pool", "SegmentIds", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("segment_pool", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = segment_pool_dygraph_function(X, SegmentIds, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sync_batch_norm(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sync_batch_norm", "X", args, 0, false);
    auto Scale =
        GetEagerTensorFromArgs("sync_batch_norm", "Scale", args, 1, false);
    auto Bias =
        GetEagerTensorFromArgs("sync_batch_norm", "Bias", args, 2, false);
    auto Mean =
        GetEagerTensorFromArgs("sync_batch_norm", "Mean", args, 3, false);
    auto Variance =
        GetEagerTensorFromArgs("sync_batch_norm", "Variance", args, 4, false);
    auto MeanOut =
        GetEagerTensorFromArgs("sync_batch_norm", "MeanOut", args, 5, false);
    auto VarianceOut = GetEagerTensorFromArgs("sync_batch_norm", "VarianceOut",
                                              args, 6, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sync_batch_norm", args, 7,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        sync_batch_norm_dygraph_function(X, Scale, Bias, Mean, Variance, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fusion_repeated_fc_relu(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("fusion_repeated_fc_relu", "X", args, 0, false);
    auto W = GetEagerTensorListFromArgs("fusion_repeated_fc_relu", "W", args, 1,
                                        false);
    auto Bias = GetEagerTensorListFromArgs("fusion_repeated_fc_relu", "Bias",
                                           args, 2, false);
    auto ReluOutNum = GetUnsignedLongFromArgs("fusion_repeated_fc_relu",
                                              "ReluOutNum", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fusion_repeated_fc_relu", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        fusion_repeated_fc_relu_dygraph_function(X, W, Bias, ReluOutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_nop(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("nop", "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs("nop", "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("nop", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = nop_dygraph_function(X, OutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_attention(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fused_attention", "X", args, 0, false);
    auto LnScale =
        GetEagerTensorFromArgs("fused_attention", "LnScale", args, 1, true);
    auto LnBias =
        GetEagerTensorFromArgs("fused_attention", "LnBias", args, 2, true);
    auto QKVW =
        GetEagerTensorFromArgs("fused_attention", "QKVW", args, 3, false);
    auto QKVBias =
        GetEagerTensorFromArgs("fused_attention", "QKVBias", args, 4, true);
    auto SrcMask =
        GetEagerTensorFromArgs("fused_attention", "SrcMask", args, 5, true);
    auto OutLinearW =
        GetEagerTensorFromArgs("fused_attention", "OutLinearW", args, 6, false);
    auto OutLinearBias = GetEagerTensorFromArgs("fused_attention",
                                                "OutLinearBias", args, 7, true);
    auto Ln2Scale =
        GetEagerTensorFromArgs("fused_attention", "Ln2Scale", args, 8, true);
    auto Ln2Bias =
        GetEagerTensorFromArgs("fused_attention", "Ln2Bias", args, 9, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_attention", args, 10,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_attention_dygraph_function(X, QKVW, OutLinearW, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_expand_as_v2(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("expand_as_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("expand_as_v2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = expand_as_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_filter_by_instag(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Ins =
        GetEagerTensorFromArgs("filter_by_instag", "Ins", args, 0, false);
    auto Ins_tag =
        GetEagerTensorFromArgs("filter_by_instag", "Ins_tag", args, 1, false);
    auto Filter_tag = GetEagerTensorFromArgs("filter_by_instag", "Filter_tag",
                                             args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("filter_by_instag", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        filter_by_instag_dygraph_function(Ins, Ins_tag, Filter_tag, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_diag_v2(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("diag_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("diag_v2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = diag_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pull_box_sparse(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Ids =
        GetEagerTensorListFromArgs("pull_box_sparse", "Ids", args, 0, false);
    auto OutNum =
        GetUnsignedLongFromArgs("pull_box_sparse", "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pull_box_sparse", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = pull_box_sparse_dygraph_function(Ids, OutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_nll_loss(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("nll_loss", "X", args, 0, false);
    auto Label = GetEagerTensorFromArgs("nll_loss", "Label", args, 1, false);
    auto Weight = GetEagerTensorFromArgs("nll_loss", "Weight", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("nll_loss", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = nll_loss_dygraph_function(X, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_dot(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("dot", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("dot", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("dot", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = dot_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_scale(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("scale", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("scale", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = scale_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_ncclBcast(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("ncclBcast", "X", args, 0, false);
    auto Communicator =
        GetEagerTensorFromArgs("ncclBcast", "Communicator", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("ncclBcast", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = ncclBcast_dygraph_function(X, Communicator, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_shuffle_batch(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("shuffle_batch", "X", args, 0, false);
    auto Seed = GetEagerTensorFromArgs("shuffle_batch", "Seed", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("shuffle_batch", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = shuffle_batch_dygraph_function(X, Seed, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_ncclReduce(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("ncclReduce", "X", args, 0, false);
    auto Communicator =
        GetEagerTensorFromArgs("ncclReduce", "Communicator", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("ncclReduce", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = ncclReduce_dygraph_function(X, Communicator, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_diag(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Diagonal = GetEagerTensorFromArgs("diag", "Diagonal", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("diag", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = diag_dygraph_function(Diagonal, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_multiplex(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Ids = GetEagerTensorFromArgs("multiplex", "Ids", args, 0, false);
    auto X = GetEagerTensorListFromArgs("multiplex", "X", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("multiplex", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = multiplex_dygraph_function(Ids, X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_leaky_relu(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("leaky_relu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("leaky_relu", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = leaky_relu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_allclose(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("allclose", "Input", args, 0, false);
    auto Other = GetEagerTensorFromArgs("allclose", "Other", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("allclose", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = allclose_dygraph_function(Input, Other, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_adamw(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("adamw", "Param", args, 0, false);
    auto Grad = GetEagerTensorFromArgs("adamw", "Grad", args, 1, false);
    auto LearningRate =
        GetEagerTensorFromArgs("adamw", "LearningRate", args, 2, false);
    auto Moment1 = GetEagerTensorFromArgs("adamw", "Moment1", args, 3, false);
    auto Moment2 = GetEagerTensorFromArgs("adamw", "Moment2", args, 4, false);
    auto Beta1Pow = GetEagerTensorFromArgs("adamw", "Beta1Pow", args, 5, false);
    auto Beta2Pow = GetEagerTensorFromArgs("adamw", "Beta2Pow", args, 6, false);
    auto MasterParam =
        GetEagerTensorFromArgs("adamw", "MasterParam", args, 7, true);
    auto ParamOut = GetEagerTensorFromArgs("adamw", "ParamOut", args, 8, false);
    auto Moment1Out =
        GetEagerTensorFromArgs("adamw", "Moment1Out", args, 9, false);
    auto Moment2Out =
        GetEagerTensorFromArgs("adamw", "Moment2Out", args, 10, false);
    auto Beta1PowOut =
        GetEagerTensorFromArgs("adamw", "Beta1PowOut", args, 11, false);
    auto Beta2PowOut =
        GetEagerTensorFromArgs("adamw", "Beta2PowOut", args, 12, false);
    auto MasterParamOut =
        GetEagerTensorFromArgs("adamw", "MasterParamOut", args, 13, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("adamw", args, 14, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = adamw_dygraph_function(Param, Grad, LearningRate, Moment1,
                                      Moment2, Beta1Pow, Beta2Pow, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elementwise_pow(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("elementwise_pow", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("elementwise_pow", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elementwise_pow", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elementwise_pow_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_prior_box(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("prior_box", "Input", args, 0, false);
    auto Image = GetEagerTensorFromArgs("prior_box", "Image", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("prior_box", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = prior_box_dygraph_function(Input, Image, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_p_norm(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("p_norm", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("p_norm", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = p_norm_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_unique_consecutive(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("unique_consecutive", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("unique_consecutive", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = unique_consecutive_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lod_reset(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("lod_reset", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lod_reset", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = lod_reset_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pad(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("pad", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pad", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = pad_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_conv(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_conv", "X", args, 0, false);
    auto Filter =
        GetEagerTensorFromArgs("sequence_conv", "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_conv", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_conv_dygraph_function(X, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_log10(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("log10", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("log10", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = log10_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_set_value(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("set_value", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("set_value", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = set_value_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bitwise_xor(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bitwise_xor", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("bitwise_xor", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bitwise_xor", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = bitwise_xor_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_center_loss(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("center_loss", "X", args, 0, false);
    auto Label = GetEagerTensorFromArgs("center_loss", "Label", args, 1, false);
    auto Centers =
        GetEagerTensorFromArgs("center_loss", "Centers", args, 2, false);
    auto CenterUpdateRate = GetEagerTensorFromArgs(
        "center_loss", "CenterUpdateRate", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("center_loss", args, 4, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = center_loss_dygraph_function(X, Label, Centers, CenterUpdateRate,
                                            attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_randint(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("randint", args, 0, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = randint_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_attention_lstm(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("attention_lstm", "X", args, 0, false);
    auto C0 = GetEagerTensorFromArgs("attention_lstm", "C0", args, 1, false);
    auto AttentionWeight = GetEagerTensorFromArgs(
        "attention_lstm", "AttentionWeight", args, 2, false);
    auto LSTMWeight =
        GetEagerTensorFromArgs("attention_lstm", "LSTMWeight", args, 3, false);
    auto LSTMBias =
        GetEagerTensorFromArgs("attention_lstm", "LSTMBias", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("attention_lstm", args, 5,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = attention_lstm_dygraph_function(X, C0, AttentionWeight,
                                               LSTMWeight, LSTMBias, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_uniform_random(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("uniform_random", args, 0,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = uniform_random_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_slice(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("slice", "Input", args, 0, false);
    auto StartsTensor =
        GetEagerTensorFromArgs("slice", "StartsTensor", args, 1, true);
    auto EndsTensor =
        GetEagerTensorFromArgs("slice", "EndsTensor", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("slice", args, 3, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = slice_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_meshgrid(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("meshgrid", "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs("meshgrid", "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("meshgrid", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = meshgrid_dygraph_function(X, OutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_hard_swish(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("hard_swish", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("hard_swish", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = hard_swish_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sin(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sin", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sin", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sin_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_mean_iou(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Predictions =
        GetEagerTensorFromArgs("mean_iou", "Predictions", args, 0, false);
    auto Labels = GetEagerTensorFromArgs("mean_iou", "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("mean_iou", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = mean_iou_dygraph_function(Predictions, Labels, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pad2d(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("pad2d", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pad2d", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = pad2d_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_inverse(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("inverse", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("inverse", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = inverse_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_spectral_norm(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Weight =
        GetEagerTensorFromArgs("spectral_norm", "Weight", args, 0, false);
    auto U = GetEagerTensorFromArgs("spectral_norm", "U", args, 1, false);
    auto V = GetEagerTensorFromArgs("spectral_norm", "V", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("spectral_norm", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = spectral_norm_dygraph_function(Weight, U, V, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_shuffle_channel(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("shuffle_channel", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("shuffle_channel", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = shuffle_channel_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_psroi_pool(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("psroi_pool", "X", args, 0, false);
    auto ROIs = GetEagerTensorFromArgs("psroi_pool", "ROIs", args, 1, false);
    auto RoisNum =
        GetEagerTensorFromArgs("psroi_pool", "RoisNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("psroi_pool", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = psroi_pool_dygraph_function(X, ROIs, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_seed(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("seed", args, 0, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = seed_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_ceil(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("ceil", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("ceil", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = ceil_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_eig(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("eig", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("eig", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = eig_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reduce_min(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reduce_min", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reduce_min", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reduce_min_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cos(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cos", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cos", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = cos_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_ncclAllReduce(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("ncclAllReduce", "X", args, 0, false);
    auto Communicator =
        GetEagerTensorFromArgs("ncclAllReduce", "Communicator", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("ncclAllReduce", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = ncclAllReduce_dygraph_function(X, Communicator, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cudnn_lstm(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("cudnn_lstm", "Input", args, 0, false);
    auto InitH = GetEagerTensorFromArgs("cudnn_lstm", "InitH", args, 1, false);
    auto InitC = GetEagerTensorFromArgs("cudnn_lstm", "InitC", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cudnn_lstm", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = cudnn_lstm_dygraph_function(Input, InitH, InitC, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reduce_sum(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reduce_sum", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reduce_sum", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reduce_sum_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_digamma(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("digamma", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("digamma", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = digamma_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_assign_value(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("assign_value", args, 0, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = assign_value_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_increment(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("increment", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("increment", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = increment_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_tdm_sampler(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("tdm_sampler", "X", args, 0, false);
    auto Travel =
        GetEagerTensorFromArgs("tdm_sampler", "Travel", args, 1, false);
    auto Layer = GetEagerTensorFromArgs("tdm_sampler", "Layer", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("tdm_sampler", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = tdm_sampler_dygraph_function(X, Travel, Layer, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_softmax_mask(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fused_softmax_mask", "X", args, 0, false);
    auto Mask =
        GetEagerTensorFromArgs("fused_softmax_mask", "Mask", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_softmax_mask", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_softmax_mask_dygraph_function(X, Mask, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_reverse(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_reverse", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_reverse", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_reverse_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_eigvalsh(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("eigvalsh", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("eigvalsh", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = eigvalsh_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_diagonal(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("diagonal", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("diagonal", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = diagonal_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_trunc(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("trunc", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("trunc", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = trunc_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_log2(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("log2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("log2", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = log2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_marker(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("marker", args, 0, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = marker_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    Py_INCREF(Py_None);
    return Py_None;
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_tanh(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("tanh", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("tanh", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = tanh_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_yolov3_loss(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("yolov3_loss", "X", args, 0, false);
    auto GTBox = GetEagerTensorFromArgs("yolov3_loss", "GTBox", args, 1, false);
    auto GTLabel =
        GetEagerTensorFromArgs("yolov3_loss", "GTLabel", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("yolov3_loss", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = yolov3_loss_dygraph_function(X, GTBox, GTLabel, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_graph_send_recv(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("graph_send_recv", "X", args, 0, false);
    auto Src_index =
        GetEagerTensorFromArgs("graph_send_recv", "Src_index", args, 1, false);
    auto Dst_index =
        GetEagerTensorFromArgs("graph_send_recv", "Dst_index", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("graph_send_recv", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = graph_send_recv_dygraph_function(X, Src_index, Dst_index, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_accuracy(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Out = GetEagerTensorFromArgs("accuracy", "Out", args, 0, false);
    auto Indices =
        GetEagerTensorFromArgs("accuracy", "Indices", args, 1, false);
    auto Label = GetEagerTensorFromArgs("accuracy", "Label", args, 2, false);
    auto Correct =
        GetEagerTensorFromArgs("accuracy", "Correct", args, 3, false);
    auto Total = GetEagerTensorFromArgs("accuracy", "Total", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("accuracy", args, 5, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = accuracy_dygraph_function(Out, Indices, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_atan(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("atan", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("atan", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = atan_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_less_than(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("less_than", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("less_than", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("less_than", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = less_than_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_unsqueeze(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("unsqueeze", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("unsqueeze", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = unsqueeze_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_crf_decoding(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Emission =
        GetEagerTensorFromArgs("crf_decoding", "Emission", args, 0, false);
    auto Transition =
        GetEagerTensorFromArgs("crf_decoding", "Transition", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("crf_decoding", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = crf_decoding_dygraph_function(Emission, Transition, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_log_softmax(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("log_softmax", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("log_softmax", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = log_softmax_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_ftrl(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("ftrl", "Param", args, 0, false);
    auto SquaredAccumulator =
        GetEagerTensorFromArgs("ftrl", "SquaredAccumulator", args, 1, false);
    auto LinearAccumulator =
        GetEagerTensorFromArgs("ftrl", "LinearAccumulator", args, 2, false);
    auto Grad = GetEagerTensorFromArgs("ftrl", "Grad", args, 3, false);
    auto LearningRate =
        GetEagerTensorFromArgs("ftrl", "LearningRate", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("ftrl", args, 5, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        ftrl_dygraph_function(Param, SquaredAccumulator, LinearAccumulator,
                              Grad, LearningRate, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_matrix_nms(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto BBoxes =
        GetEagerTensorFromArgs("matrix_nms", "BBoxes", args, 0, false);
    auto Scores =
        GetEagerTensorFromArgs("matrix_nms", "Scores", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("matrix_nms", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = matrix_nms_dygraph_function(BBoxes, Scores, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_top_k_v2(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("top_k_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("top_k_v2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = top_k_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cast(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cast", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cast", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = cast_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_tanh_shrink(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("tanh_shrink", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("tanh_shrink", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = tanh_shrink_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_hard_shrink(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("hard_shrink", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("hard_shrink", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = hard_shrink_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_multiclass_nms(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto BBoxes =
        GetEagerTensorFromArgs("multiclass_nms", "BBoxes", args, 0, false);
    auto Scores =
        GetEagerTensorFromArgs("multiclass_nms", "Scores", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("multiclass_nms", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = multiclass_nms_dygraph_function(BBoxes, Scores, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fusion_transpose_flatten_concat(PyObject *self,
                                                           PyObject *args,
                                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("fusion_transpose_flatten_concat", "X",
                                        args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fusion_transpose_flatten_concat", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fusion_transpose_flatten_concat_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_unpad(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_unpad", "X", args, 0, false);
    auto Length =
        GetEagerTensorFromArgs("sequence_unpad", "Length", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_unpad", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_unpad_dygraph_function(X, Length, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_elemwise_add_activation(PyObject *self,
                                                         PyObject *args,
                                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fused_elemwise_add_activation", "X", args,
                                    0, false);
    auto Y = GetEagerTensorFromArgs("fused_elemwise_add_activation", "Y", args,
                                    1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_elemwise_add_activation", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_elemwise_add_activation_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pull_sparse_v2(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Ids =
        GetEagerTensorListFromArgs("pull_sparse_v2", "Ids", args, 0, false);
    auto W = GetEagerTensorListFromArgs("pull_sparse_v2", "W", args, 1, false);
    auto OutNum =
        GetUnsignedLongFromArgs("pull_sparse_v2", "OutNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pull_sparse_v2", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = pull_sparse_v2_dygraph_function(Ids, W, OutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_frobenius_norm(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("frobenius_norm", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("frobenius_norm", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = frobenius_norm_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_crop(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("crop", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("crop", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = crop_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cross_entropy2(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cross_entropy2", "X", args, 0, false);
    auto Label =
        GetEagerTensorFromArgs("cross_entropy2", "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cross_entropy2", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = cross_entropy2_dygraph_function(X, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_skip_layernorm(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("skip_layernorm", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("skip_layernorm", "Y", args, 1, false);
    auto Scale =
        GetEagerTensorFromArgs("skip_layernorm", "Scale", args, 2, false);
    auto Bias =
        GetEagerTensorFromArgs("skip_layernorm", "Bias", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("skip_layernorm", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = skip_layernorm_dygraph_function(X, Y, Scale, Bias, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_tdm_child(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("tdm_child", "X", args, 0, false);
    auto TreeInfo =
        GetEagerTensorFromArgs("tdm_child", "TreeInfo", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("tdm_child", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = tdm_child_dygraph_function(X, TreeInfo, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_embedding_seq_pool(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto W =
        GetEagerTensorFromArgs("fused_embedding_seq_pool", "W", args, 0, false);
    auto Ids = GetEagerTensorFromArgs("fused_embedding_seq_pool", "Ids", args,
                                      1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_embedding_seq_pool", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_embedding_seq_pool_dygraph_function(W, Ids, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_erf(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("erf", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("erf", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = erf_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_conv2d_inception_fusion(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("conv2d_inception_fusion", "Input",
                                        args, 0, false);
    auto Filter = GetEagerTensorListFromArgs("conv2d_inception_fusion",
                                             "Filter", args, 1, false);
    auto Bias = GetEagerTensorListFromArgs("conv2d_inception_fusion", "Bias",
                                           args, 2, false);
    auto TempOutputNum = GetUnsignedLongFromArgs(
        "conv2d_inception_fusion", "TempOutputNum", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("conv2d_inception_fusion", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = conv2d_inception_fusion_dygraph_function(Input, Filter, Bias,
                                                        TempOutputNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_trilinear_interp(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("trilinear_interp", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("trilinear_interp", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = trilinear_interp_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_logsumexp(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("logsumexp", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("logsumexp", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = logsumexp_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fusion_seqpool_concat(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("fusion_seqpool_concat", "X", args, 0,
                                        false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fusion_seqpool_concat", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fusion_seqpool_concat_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_alloc_float_status(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("alloc_float_status", args, 0,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = alloc_float_status_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_concat(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("sequence_concat", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_concat", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_concat_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fusion_seqpool_cvm_concat(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("fusion_seqpool_cvm_concat", "X", args,
                                        0, false);
    auto CVM = GetEagerTensorFromArgs("fusion_seqpool_cvm_concat", "CVM", args,
                                      1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fusion_seqpool_cvm_concat", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fusion_seqpool_cvm_concat_dygraph_function(X, CVM, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_similarity_focus(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("similarity_focus", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("similarity_focus", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = similarity_focus_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_argsort(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("argsort", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("argsort", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = argsort_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_expand(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_expand", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("sequence_expand", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_expand", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_expand_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sgd(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("sgd", "Param", args, 0, false);
    auto LearningRate =
        GetEagerTensorFromArgs("sgd", "LearningRate", args, 1, false);
    auto Grad = GetEagerTensorFromArgs("sgd", "Grad", args, 2, false);
    auto ParamOut = GetEagerTensorFromArgs("sgd", "ParamOut", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sgd", args, 4, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sgd_dygraph_function(Param, LearningRate, Grad, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_bn_add_activation(PyObject *self,
                                                   PyObject *args,
                                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("fused_bn_add_activation", "X", args, 0, false);
    auto Z =
        GetEagerTensorFromArgs("fused_bn_add_activation", "Z", args, 1, false);
    auto Scale = GetEagerTensorFromArgs("fused_bn_add_activation", "Scale",
                                        args, 2, false);
    auto Bias = GetEagerTensorFromArgs("fused_bn_add_activation", "Bias", args,
                                       3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_bn_add_activation", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        fused_bn_add_activation_dygraph_function(X, Z, Scale, Bias, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bilinear_interp_v2(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bilinear_interp_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bilinear_interp_v2", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = bilinear_interp_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_clip(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("clip", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("clip", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = clip_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_deformable_conv_v1(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("deformable_conv_v1", "Input", args, 0, false);
    auto Offset =
        GetEagerTensorFromArgs("deformable_conv_v1", "Offset", args, 1, false);
    auto Filter =
        GetEagerTensorFromArgs("deformable_conv_v1", "Filter", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("deformable_conv_v1", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        deformable_conv_v1_dygraph_function(Input, Offset, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_hinge_loss(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Logits =
        GetEagerTensorFromArgs("hinge_loss", "Logits", args, 0, false);
    auto Labels =
        GetEagerTensorFromArgs("hinge_loss", "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("hinge_loss", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = hinge_loss_dygraph_function(Logits, Labels, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_determinant(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("determinant", "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("determinant", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = determinant_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_conv2d_transpose(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("conv2d_transpose", "Input", args, 0, false);
    auto Filter =
        GetEagerTensorFromArgs("conv2d_transpose", "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("conv2d_transpose", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = conv2d_transpose_dygraph_function(Input, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_memcpy_d2h(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("memcpy_d2h", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("memcpy_d2h", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = memcpy_d2h_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_softsign(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("softsign", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("softsign", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = softsign_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fake_quantize_dequantize_abs_max(PyObject *self,
                                                            PyObject *args,
                                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fake_quantize_dequantize_abs_max", "X",
                                    args, 0, false);
    auto Out = GetEagerTensorFromArgs("fake_quantize_dequantize_abs_max", "Out",
                                      args, 1, false);
    auto OutScale = GetEagerTensorFromArgs("fake_quantize_dequantize_abs_max",
                                           "OutScale", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fake_quantize_dequantize_abs_max", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fake_quantize_dequantize_abs_max_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_broadcast_tensors(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorListFromArgs("broadcast_tensors", "X", args, 0, false);
    auto OutNum =
        GetUnsignedLongFromArgs("broadcast_tensors", "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("broadcast_tensors", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = broadcast_tensors_dygraph_function(X, OutNum, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_grid_sampler(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("grid_sampler", "X", args, 0, false);
    auto Grid = GetEagerTensorFromArgs("grid_sampler", "Grid", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("grid_sampler", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = grid_sampler_dygraph_function(X, Grid, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fft_c2r(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fft_c2r", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fft_c2r", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fft_c2r_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pyramid_hash(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("pyramid_hash", "X", args, 0, false);
    auto W = GetEagerTensorFromArgs("pyramid_hash", "W", args, 1, false);
    auto WhiteList =
        GetEagerTensorFromArgs("pyramid_hash", "WhiteList", args, 2, false);
    auto BlackList =
        GetEagerTensorFromArgs("pyramid_hash", "BlackList", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pyramid_hash", args, 4, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = pyramid_hash_dygraph_function(X, W, WhiteList, BlackList, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fake_quantize_dequantize_moving_average_abs_max(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs(
        "fake_quantize_dequantize_moving_average_abs_max", "X", args, 0, false);
    auto InScale = GetEagerTensorFromArgs(
        "fake_quantize_dequantize_moving_average_abs_max", "InScale", args, 1,
        false);
    auto InAccum = GetEagerTensorFromArgs(
        "fake_quantize_dequantize_moving_average_abs_max", "InAccum", args, 2,
        true);
    auto InState = GetEagerTensorFromArgs(
        "fake_quantize_dequantize_moving_average_abs_max", "InState", args, 3,
        true);
    auto Out = GetEagerTensorFromArgs(
        "fake_quantize_dequantize_moving_average_abs_max", "Out", args, 4,
        false);
    auto OutScale = GetEagerTensorFromArgs(
        "fake_quantize_dequantize_moving_average_abs_max", "OutScale", args, 5,
        false);
    auto OutState = GetEagerTensorFromArgs(
        "fake_quantize_dequantize_moving_average_abs_max", "OutState", args, 6,
        true);
    auto OutAccum = GetEagerTensorFromArgs(
        "fake_quantize_dequantize_moving_average_abs_max", "OutAccum", args, 7,
        true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(
        "fake_quantize_dequantize_moving_average_abs_max", args, 8,
        PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fake_quantize_dequantize_moving_average_abs_max_dygraph_function(
        X, InScale, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_multi_dot(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("multi_dot", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("multi_dot", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = multi_dot_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_pool(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_pool", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_pool", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_pool_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_transpose(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("transpose", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("transpose", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = transpose_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_top_k(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("top_k", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("top_k", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = top_k_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_dist(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("dist", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("dist", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("dist", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = dist_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_affine_grid(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Theta = GetEagerTensorFromArgs("affine_grid", "Theta", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("affine_grid", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = affine_grid_dygraph_function(Theta, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_gaussian_random_batch_size_like(PyObject *self,
                                                           PyObject *args,
                                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("gaussian_random_batch_size_like",
                                        "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("gaussian_random_batch_size_like", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = gaussian_random_batch_size_like_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fake_channel_wise_dequantize_max_abs(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fake_channel_wise_dequantize_max_abs", "X",
                                    args, 0, false);
    auto Scales = GetEagerTensorListFromArgs(
        "fake_channel_wise_dequantize_max_abs", "Scales", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fake_channel_wise_dequantize_max_abs", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out =
        fake_channel_wise_dequantize_max_abs_dygraph_function(X, Scales, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reciprocal(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reciprocal", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reciprocal", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reciprocal_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_mask(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_mask", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_mask", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_mask_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fill_diagonal_tensor(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("fill_diagonal_tensor", "X", args, 0, false);
    auto Y =
        GetEagerTensorFromArgs("fill_diagonal_tensor", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fill_diagonal_tensor", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fill_diagonal_tensor_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_abs(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("abs", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("abs", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = abs_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_partial_concat(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("partial_concat", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("partial_concat", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = partial_concat_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elu(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("elu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elu", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_index_select(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("index_select", "X", args, 0, false);
    auto Index =
        GetEagerTensorFromArgs("index_select", "Index", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("index_select", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = index_select_dygraph_function(X, Index, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_row_conv(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("row_conv", "X", args, 0, false);
    auto Filter = GetEagerTensorFromArgs("row_conv", "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("row_conv", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = row_conv_dygraph_function(X, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cross(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cross", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("cross", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cross", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = cross_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elementwise_mul(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("elementwise_mul", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("elementwise_mul", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elementwise_mul", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elementwise_mul_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_decayed_adagrad(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param =
        GetEagerTensorFromArgs("decayed_adagrad", "Param", args, 0, false);
    auto Grad =
        GetEagerTensorFromArgs("decayed_adagrad", "Grad", args, 1, false);
    auto Moment =
        GetEagerTensorFromArgs("decayed_adagrad", "Moment", args, 2, false);
    auto LearningRate = GetEagerTensorFromArgs("decayed_adagrad",
                                               "LearningRate", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("decayed_adagrad", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = decayed_adagrad_dygraph_function(Param, Grad, Moment,
                                                LearningRate, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bipartite_match(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto DistMat =
        GetEagerTensorFromArgs("bipartite_match", "DistMat", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bipartite_match", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = bipartite_match_dygraph_function(DistMat, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_run_program(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("run_program", "X", args, 0, false);
    auto Params =
        GetEagerTensorListFromArgs("run_program", "Params", args, 1, true);
    auto Out = GetEagerTensorListFromArgs("run_program", "Out", args, 2, false);
    auto OutScope =
        GetEagerTensorFromArgs("run_program", "OutScope", args, 3, false);
    auto DOut =
        GetEagerTensorListFromArgs("run_program", "DOut", args, 4, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("run_program", args, 5, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = run_program_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fake_quantize_moving_average_abs_max(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fake_quantize_moving_average_abs_max", "X",
                                    args, 0, false);
    auto InScale = GetEagerTensorFromArgs(
        "fake_quantize_moving_average_abs_max", "InScale", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fake_quantize_moving_average_abs_max", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fake_quantize_moving_average_abs_max_dygraph_function(X, InScale,
                                                                     attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_mine_hard_examples(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto ClsLoss =
        GetEagerTensorFromArgs("mine_hard_examples", "ClsLoss", args, 0, false);
    auto MatchIndices = GetEagerTensorFromArgs("mine_hard_examples",
                                               "MatchIndices", args, 1, false);
    auto MatchDist = GetEagerTensorFromArgs("mine_hard_examples", "MatchDist",
                                            args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("mine_hard_examples", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = mine_hard_examples_dygraph_function(ClsLoss, MatchIndices,
                                                   MatchDist, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_target_assign(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("target_assign", "X", args, 0, false);
    auto MatchIndices =
        GetEagerTensorFromArgs("target_assign", "MatchIndices", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("target_assign", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = target_assign_dygraph_function(X, MatchIndices, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lstm(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("lstm", "Input", args, 0, false);
    auto Weight = GetEagerTensorFromArgs("lstm", "Weight", args, 1, false);
    auto Bias = GetEagerTensorFromArgs("lstm", "Bias", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lstm", args, 3, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = lstm_dygraph_function(Input, Weight, Bias, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_truncated_gaussian_random(PyObject *self,
                                                     PyObject *args,
                                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("truncated_gaussian_random", args, 0,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = truncated_gaussian_random_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_match_matrix_tensor(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("match_matrix_tensor", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("match_matrix_tensor", "Y", args, 1, false);
    auto W = GetEagerTensorFromArgs("match_matrix_tensor", "W", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("match_matrix_tensor", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = match_matrix_tensor_dygraph_function(X, Y, W, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elementwise_div(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("elementwise_div", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("elementwise_div", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elementwise_div", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elementwise_div_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_kldiv_loss(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("kldiv_loss", "X", args, 0, false);
    auto Target =
        GetEagerTensorFromArgs("kldiv_loss", "Target", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("kldiv_loss", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = kldiv_loss_dygraph_function(X, Target, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cumsum(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cumsum", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cumsum", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = cumsum_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sum(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorListFromArgs("sum", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sum", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sum_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_proximal_adagrad(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param =
        GetEagerTensorFromArgs("proximal_adagrad", "Param", args, 0, false);
    auto Moment =
        GetEagerTensorFromArgs("proximal_adagrad", "Moment", args, 1, false);
    auto Grad =
        GetEagerTensorFromArgs("proximal_adagrad", "Grad", args, 2, false);
    auto LearningRate = GetEagerTensorFromArgs("proximal_adagrad",
                                               "LearningRate", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("proximal_adagrad", args, 4,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = proximal_adagrad_dygraph_function(Param, Moment, Grad,
                                                 LearningRate, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_update_loss_scaling(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorListFromArgs("update_loss_scaling", "X", args, 0, false);
    auto FoundInfinite = GetEagerTensorFromArgs(
        "update_loss_scaling", "FoundInfinite", args, 1, false);
    auto PrevLossScaling = GetEagerTensorFromArgs(
        "update_loss_scaling", "PrevLossScaling", args, 2, false);
    auto InGoodSteps = GetEagerTensorFromArgs("update_loss_scaling",
                                              "InGoodSteps", args, 3, false);
    auto InBadSteps = GetEagerTensorFromArgs("update_loss_scaling",
                                             "InBadSteps", args, 4, false);
    auto Out = GetEagerTensorListFromArgs("update_loss_scaling", "Out", args, 5,
                                          false);
    auto LossScaling = GetEagerTensorFromArgs("update_loss_scaling",
                                              "LossScaling", args, 6, false);
    auto OutGoodSteps = GetEagerTensorFromArgs("update_loss_scaling",
                                               "OutGoodSteps", args, 7, false);
    auto OutBadSteps = GetEagerTensorFromArgs("update_loss_scaling",
                                              "OutBadSteps", args, 8, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("update_loss_scaling", args, 9,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = update_loss_scaling_dygraph_function(
        X, FoundInfinite, PrevLossScaling, InGoodSteps, InBadSteps, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_shard_index(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("shard_index", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("shard_index", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = shard_index_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_selu(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("selu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("selu", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = selu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_mean(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("mean", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("mean", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = mean_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_gumbel_softmax(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("gumbel_softmax", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("gumbel_softmax", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = gumbel_softmax_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sequence_pad(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sequence_pad", "X", args, 0, false);
    auto PadValue =
        GetEagerTensorFromArgs("sequence_pad", "PadValue", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sequence_pad", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = sequence_pad_dygraph_function(X, PadValue, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_tree_conv(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto NodesVector =
        GetEagerTensorFromArgs("tree_conv", "NodesVector", args, 0, false);
    auto EdgeSet =
        GetEagerTensorFromArgs("tree_conv", "EdgeSet", args, 1, false);
    auto Filter = GetEagerTensorFromArgs("tree_conv", "Filter", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("tree_conv", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = tree_conv_dygraph_function(NodesVector, EdgeSet, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_assign(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("assign", "X", args, 0, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("assign", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = assign_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_flatten_contiguous_range(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("flatten_contiguous_range", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("flatten_contiguous_range", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = flatten_contiguous_range_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_tril_triu(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("tril_triu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("tril_triu", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = tril_triu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_brelu(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("brelu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("brelu", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = brelu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_celu(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("celu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("celu", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = celu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reduce_mean(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reduce_mean", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reduce_mean", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reduce_mean_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_sinh(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("sinh", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("sinh", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = sinh_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_rank_loss(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Label = GetEagerTensorFromArgs("rank_loss", "Label", args, 0, false);
    auto Left = GetEagerTensorFromArgs("rank_loss", "Left", args, 1, false);
    auto Right = GetEagerTensorFromArgs("rank_loss", "Right", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("rank_loss", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = rank_loss_dygraph_function(Label, Left, Right, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reduce_max(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reduce_max", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reduce_max", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reduce_max_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fusion_gru(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fusion_gru", "X", args, 0, false);
    auto WeightX =
        GetEagerTensorFromArgs("fusion_gru", "WeightX", args, 1, false);
    auto WeightH =
        GetEagerTensorFromArgs("fusion_gru", "WeightH", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fusion_gru", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = fusion_gru_dygraph_function(X, WeightX, WeightH, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fill_zeros_like2(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fill_zeros_like2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fill_zeros_like2", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fill_zeros_like2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_expm1(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("expm1", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("expm1", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = expm1_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_squared_l2_norm(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("squared_l2_norm", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("squared_l2_norm", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = squared_l2_norm_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elementwise_sub(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("elementwise_sub", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("elementwise_sub", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elementwise_sub", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elementwise_sub_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_margin_rank_loss(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X1 = GetEagerTensorFromArgs("margin_rank_loss", "X1", args, 0, false);
    auto X2 = GetEagerTensorFromArgs("margin_rank_loss", "X2", args, 1, false);
    auto Label =
        GetEagerTensorFromArgs("margin_rank_loss", "Label", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("margin_rank_loss", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = margin_rank_loss_dygraph_function(X1, X2, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_faster_tokenizer(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Vocab =
        GetEagerTensorFromArgs("faster_tokenizer", "Vocab", args, 0, false);
    auto Text =
        GetEagerTensorFromArgs("faster_tokenizer", "Text", args, 1, false);
    auto TextPair =
        GetEagerTensorFromArgs("faster_tokenizer", "TextPair", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("faster_tokenizer", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = faster_tokenizer_dygraph_function(Vocab, Text, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_relu(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("relu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("relu", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = relu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_is_empty(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("is_empty", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("is_empty", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = is_empty_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_reduce_all(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("reduce_all", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("reduce_all", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = reduce_all_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_edit_distance(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Hyps = GetEagerTensorFromArgs("edit_distance", "Hyps", args, 0, false);
    auto Refs = GetEagerTensorFromArgs("edit_distance", "Refs", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("edit_distance", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = edit_distance_dygraph_function(Hyps, Refs, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_bmm(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("bmm", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("bmm", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("bmm", args, 2, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = bmm_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_yolo_box(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("yolo_box", "X", args, 0, false);
    auto ImgSize =
        GetEagerTensorFromArgs("yolo_box", "ImgSize", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("yolo_box", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = yolo_box_dygraph_function(X, ImgSize, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_soft_relu(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("soft_relu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("soft_relu", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = soft_relu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_density_prior_box(PyObject *self, PyObject *args,
                                             PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("density_prior_box", "Input", args, 0, false);
    auto Image =
        GetEagerTensorFromArgs("density_prior_box", "Image", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("density_prior_box", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = density_prior_box_dygraph_function(Input, Image, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_eye(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("eye", args, 0, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = eye_dygraph_function(attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_swish(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("swish", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("swish", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = swish_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cross_entropy(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cross_entropy", "X", args, 0, false);
    auto Label =
        GetEagerTensorFromArgs("cross_entropy", "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cross_entropy", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = cross_entropy_dygraph_function(X, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_dpsgd(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Param = GetEagerTensorFromArgs("dpsgd", "Param", args, 0, false);
    auto Grad = GetEagerTensorFromArgs("dpsgd", "Grad", args, 1, false);
    auto LearningRate =
        GetEagerTensorFromArgs("dpsgd", "LearningRate", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("dpsgd", args, 3, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = dpsgd_dygraph_function(Param, Grad, LearningRate, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_cholesky(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("cholesky", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("cholesky", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = cholesky_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_batch_fc(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("batch_fc", "Input", args, 0, false);
    auto W = GetEagerTensorFromArgs("batch_fc", "W", args, 1, false);
    auto Bias = GetEagerTensorFromArgs("batch_fc", "Bias", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("batch_fc", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = batch_fc_dygraph_function(Input, W, Bias, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_nearest_interp(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("nearest_interp", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("nearest_interp", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = nearest_interp_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_gather(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("gather", "X", args, 0, false);
    auto Index = GetEagerTensorFromArgs("gather", "Index", args, 1, false);
    auto Axis = GetEagerTensorFromArgs("gather", "Axis", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("gather", args, 3, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = gather_dygraph_function(X, Index, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_trilinear_interp_v2(PyObject *self, PyObject *args,
                                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("trilinear_interp_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("trilinear_interp_v2", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = trilinear_interp_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_box_clip(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("box_clip", "Input", args, 0, false);
    auto ImInfo = GetEagerTensorFromArgs("box_clip", "ImInfo", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("box_clip", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = box_clip_dygraph_function(Input, ImInfo, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_isnan_v2(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("isnan_v2", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("isnan_v2", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = isnan_v2_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_softmax(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("softmax", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("softmax", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = softmax_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_conv2d_fusion(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input =
        GetEagerTensorFromArgs("conv2d_fusion", "Input", args, 0, false);
    auto Filter =
        GetEagerTensorFromArgs("conv2d_fusion", "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("conv2d_fusion", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = conv2d_fusion_dygraph_function(Input, Filter, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fused_batch_norm_act(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X =
        GetEagerTensorFromArgs("fused_batch_norm_act", "X", args, 0, false);
    auto Scale =
        GetEagerTensorFromArgs("fused_batch_norm_act", "Scale", args, 1, false);
    auto Bias =
        GetEagerTensorFromArgs("fused_batch_norm_act", "Bias", args, 2, false);
    auto Mean =
        GetEagerTensorFromArgs("fused_batch_norm_act", "Mean", args, 3, false);
    auto Variance = GetEagerTensorFromArgs("fused_batch_norm_act", "Variance",
                                           args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fused_batch_norm_act", args, 5,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fused_batch_norm_act_dygraph_function(X, Scale, Bias, Mean,
                                                     Variance, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_get_float_status(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto FloatStatus = GetEagerTensorFromArgs("get_float_status", "FloatStatus",
                                              args, 0, false);
    auto FloatStatusOut = GetEagerTensorFromArgs(
        "get_float_status", "FloatStatusOut", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("get_float_status", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = get_float_status_dygraph_function(FloatStatus, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_index_sample(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("index_sample", "X", args, 0, false);
    auto Index =
        GetEagerTensorFromArgs("index_sample", "Index", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("index_sample", args, 2, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = index_sample_dygraph_function(X, Index, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_elementwise_min(PyObject *self, PyObject *args,
                                           PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("elementwise_min", "X", args, 0, false);
    auto Y = GetEagerTensorFromArgs("elementwise_min", "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("elementwise_min", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = elementwise_min_dygraph_function(X, Y, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_logical_not(PyObject *self, PyObject *args,
                                       PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("logical_not", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("logical_not", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = logical_not_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_collect_fpn_proposals(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto MultiLevelRois = GetEagerTensorListFromArgs(
        "collect_fpn_proposals", "MultiLevelRois", args, 0, false);
    auto MultiLevelScores = GetEagerTensorListFromArgs(
        "collect_fpn_proposals", "MultiLevelScores", args, 1, false);
    auto MultiLevelRoIsNum = GetEagerTensorListFromArgs(
        "collect_fpn_proposals", "MultiLevelRoIsNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("collect_fpn_proposals", args, 3,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = collect_fpn_proposals_dygraph_function(MultiLevelRois,
                                                      MultiLevelScores, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_pixel_shuffle(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("pixel_shuffle", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("pixel_shuffle", args, 1, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = pixel_shuffle_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_thresholded_relu(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("thresholded_relu", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("thresholded_relu", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = thresholded_relu_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_polygon_box_transform(PyObject *self, PyObject *args,
                                                 PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Input = GetEagerTensorFromArgs("polygon_box_transform", "Input", args,
                                        0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("polygon_box_transform", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = polygon_box_transform_dygraph_function(Input, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_lookup_table_dequant(PyObject *self, PyObject *args,
                                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto W =
        GetEagerTensorFromArgs("lookup_table_dequant", "W", args, 0, false);
    auto Ids =
        GetEagerTensorFromArgs("lookup_table_dequant", "Ids", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("lookup_table_dequant", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = lookup_table_dequant_dygraph_function(W, Ids, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_warpctc(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto Logits = GetEagerTensorFromArgs("warpctc", "Logits", args, 0, false);
    auto Label = GetEagerTensorFromArgs("warpctc", "Label", args, 1, false);
    auto LogitsLength =
        GetEagerTensorFromArgs("warpctc", "LogitsLength", args, 2, true);
    auto LabelLength =
        GetEagerTensorFromArgs("warpctc", "LabelLength", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("warpctc", args, 4, PyTuple_GET_SIZE(args),
                               attrs);
    tstate = PyEval_SaveThread();
    auto out = warpctc_dygraph_function(Logits, Label, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_fake_channel_wise_quantize_abs_max(
    PyObject *self, PyObject *args, PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("fake_channel_wise_quantize_abs_max", "X",
                                    args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("fake_channel_wise_quantize_abs_max", args, 1,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = fake_channel_wise_quantize_abs_max_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_dequantize_abs_max(PyObject *self, PyObject *args,
                                              PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("dequantize_abs_max", "X", args, 0, false);
    auto Scale =
        GetEagerTensorFromArgs("dequantize_abs_max", "Scale", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("dequantize_abs_max", args, 2,
                               PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = dequantize_abs_max_dygraph_function(X, Scale, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_svd(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("svd", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("svd", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = svd_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *eager_api_flip(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyThreadState *tstate = nullptr;
  try {
    auto X = GetEagerTensorFromArgs("flip", "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("flip", args, 1, PyTuple_GET_SIZE(args), attrs);
    tstate = PyEval_SaveThread();
    auto out = flip_dygraph_function(X, attrs);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch (...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyMethodDef ExtestMethods[] = {
    {"rsqrt", (PyCFunction)(void (*)(void))eager_api_rsqrt,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for rsqrt in dygraph."},
    {"multihead_matmul",
     (PyCFunction)(void (*)(void))eager_api_multihead_matmul,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multihead_matmul in dygraph."},
    {"addmm", (PyCFunction)(void (*)(void))eager_api_addmm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for addmm in dygraph."},
    {"gru", (PyCFunction)(void (*)(void))eager_api_gru,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gru in dygraph."},
    {"round", (PyCFunction)(void (*)(void))eager_api_round,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for round in dygraph."},
    {"rank_attention", (PyCFunction)(void (*)(void))eager_api_rank_attention,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for rank_attention in dygraph."},
    {"fused_embedding_fc_lstm",
     (PyCFunction)(void (*)(void))eager_api_fused_embedding_fc_lstm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_embedding_fc_lstm in dygraph."},
    {"where_index", (PyCFunction)(void (*)(void))eager_api_where_index,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for where_index in dygraph."},
    {"bicubic_interp", (PyCFunction)(void (*)(void))eager_api_bicubic_interp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bicubic_interp in dygraph."},
    {"arg_min", (PyCFunction)(void (*)(void))eager_api_arg_min,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for arg_min in dygraph."},
    {"tile", (PyCFunction)(void (*)(void))eager_api_tile,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tile in dygraph."},
    {"bilinear_tensor_product",
     (PyCFunction)(void (*)(void))eager_api_bilinear_tensor_product,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bilinear_tensor_product in dygraph."},
    {"ctc_align", (PyCFunction)(void (*)(void))eager_api_ctc_align,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for ctc_align in dygraph."},
    {"pow2_decay_with_linear_warmup",
     (PyCFunction)(void (*)(void))eager_api_pow2_decay_with_linear_warmup,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pow2_decay_with_linear_warmup in dygraph."},
    {"split", (PyCFunction)(void (*)(void))eager_api_split,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for split in dygraph."},
    {"fc", (PyCFunction)(void (*)(void))eager_api_fc,
     METH_VARARGS | METH_KEYWORDS, "C++ interface function for fc in dygraph."},
    {"clear_float_status",
     (PyCFunction)(void (*)(void))eager_api_clear_float_status,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for clear_float_status in dygraph."},
    {"matmul_v2", (PyCFunction)(void (*)(void))eager_api_matmul_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matmul_v2 in dygraph."},
    {"load", (PyCFunction)(void (*)(void))eager_api_load,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for load in dygraph."},
    {"elementwise_max", (PyCFunction)(void (*)(void))eager_api_elementwise_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_max in dygraph."},
    {"adadelta", (PyCFunction)(void (*)(void))eager_api_adadelta,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for adadelta in dygraph."},
    {"chunk_eval", (PyCFunction)(void (*)(void))eager_api_chunk_eval,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for chunk_eval in dygraph."},
    {"check_finite_and_unscale",
     (PyCFunction)(void (*)(void))eager_api_check_finite_and_unscale,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for check_finite_and_unscale in dygraph."},
    {"sparse_momentum", (PyCFunction)(void (*)(void))eager_api_sparse_momentum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sparse_momentum in dygraph."},
    {"tan", (PyCFunction)(void (*)(void))eager_api_tan,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tan in dygraph."},
    {"adam", (PyCFunction)(void (*)(void))eager_api_adam,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for adam in dygraph."},
    {"fsp", (PyCFunction)(void (*)(void))eager_api_fsp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fsp in dygraph."},
    {"where", (PyCFunction)(void (*)(void))eager_api_where,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for where in dygraph."},
    {"logical_xor", (PyCFunction)(void (*)(void))eager_api_logical_xor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logical_xor in dygraph."},
    {"multiclass_nms3", (PyCFunction)(void (*)(void))eager_api_multiclass_nms3,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multiclass_nms3 in dygraph."},
    {"one_hot_v2", (PyCFunction)(void (*)(void))eager_api_one_hot_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for one_hot_v2 in dygraph."},
    {"sequence_softmax",
     (PyCFunction)(void (*)(void))eager_api_sequence_softmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_softmax in dygraph."},
    {"affine_channel", (PyCFunction)(void (*)(void))eager_api_affine_channel,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for affine_channel in dygraph."},
    {"triangular_solve",
     (PyCFunction)(void (*)(void))eager_api_triangular_solve,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for triangular_solve in dygraph."},
    {"sequence_topk_avg_pooling",
     (PyCFunction)(void (*)(void))eager_api_sequence_topk_avg_pooling,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_topk_avg_pooling in dygraph."},
    {"space_to_depth", (PyCFunction)(void (*)(void))eager_api_space_to_depth,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for space_to_depth in dygraph."},
    {"reverse", (PyCFunction)(void (*)(void))eager_api_reverse,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reverse in dygraph."},
    {"fused_embedding_eltwise_layernorm",
     (PyCFunction)(void (*)(void))eager_api_fused_embedding_eltwise_layernorm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_embedding_eltwise_layernorm in "
     "dygraph."},
    {"expand_v2", (PyCFunction)(void (*)(void))eager_api_expand_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for expand_v2 in dygraph."},
    {"lgamma", (PyCFunction)(void (*)(void))eager_api_lgamma,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lgamma in dygraph."},
    {"solve", (PyCFunction)(void (*)(void))eager_api_solve,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for solve in dygraph."},
    {"deformable_psroi_pooling",
     (PyCFunction)(void (*)(void))eager_api_deformable_psroi_pooling,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for deformable_psroi_pooling in dygraph."},
    {"instance_norm", (PyCFunction)(void (*)(void))eager_api_instance_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for instance_norm in dygraph."},
    {"decode_jpeg", (PyCFunction)(void (*)(void))eager_api_decode_jpeg,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for decode_jpeg in dygraph."},
    {"gather_nd", (PyCFunction)(void (*)(void))eager_api_gather_nd,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gather_nd in dygraph."},
    {"reduce_prod", (PyCFunction)(void (*)(void))eager_api_reduce_prod,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reduce_prod in dygraph."},
    {"matrix_rank", (PyCFunction)(void (*)(void))eager_api_matrix_rank,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matrix_rank in dygraph."},
    {"asin", (PyCFunction)(void (*)(void))eager_api_asin,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for asin in dygraph."},
    {"lstmp", (PyCFunction)(void (*)(void))eager_api_lstmp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lstmp in dygraph."},
    {"iou_similarity", (PyCFunction)(void (*)(void))eager_api_iou_similarity,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for iou_similarity in dygraph."},
    {"huber_loss", (PyCFunction)(void (*)(void))eager_api_huber_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for huber_loss in dygraph."},
    {"one_hot", (PyCFunction)(void (*)(void))eager_api_one_hot,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for one_hot in dygraph."},
    {"sequence_slice", (PyCFunction)(void (*)(void))eager_api_sequence_slice,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_slice in dygraph."},
    {"lookup_table", (PyCFunction)(void (*)(void))eager_api_lookup_table,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lookup_table in dygraph."},
    {"softplus", (PyCFunction)(void (*)(void))eager_api_softplus,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for softplus in dygraph."},
    {"depthwise_conv2d",
     (PyCFunction)(void (*)(void))eager_api_depthwise_conv2d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for depthwise_conv2d in dygraph."},
    {"fused_fc_elementwise_layernorm",
     (PyCFunction)(void (*)(void))eager_api_fused_fc_elementwise_layernorm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_fc_elementwise_layernorm in dygraph."},
    {"sigmoid_cross_entropy_with_logits",
     (PyCFunction)(void (*)(void))eager_api_sigmoid_cross_entropy_with_logits,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sigmoid_cross_entropy_with_logits in "
     "dygraph."},
    {"exp", (PyCFunction)(void (*)(void))eager_api_exp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for exp in dygraph."},
    {"scatter", (PyCFunction)(void (*)(void))eager_api_scatter,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for scatter in dygraph."},
    {"equal_all", (PyCFunction)(void (*)(void))eager_api_equal_all,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for equal_all in dygraph."},
    {"searchsorted", (PyCFunction)(void (*)(void))eager_api_searchsorted,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for searchsorted in dygraph."},
    {"fusion_squared_mat_sub",
     (PyCFunction)(void (*)(void))eager_api_fusion_squared_mat_sub,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fusion_squared_mat_sub in dygraph."},
    {"unique", (PyCFunction)(void (*)(void))eager_api_unique,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unique in dygraph."},
    {"log", (PyCFunction)(void (*)(void))eager_api_log,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log in dygraph."},
    {"conv_shift", (PyCFunction)(void (*)(void))eager_api_conv_shift,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv_shift in dygraph."},
    {"smooth_l1_loss", (PyCFunction)(void (*)(void))eager_api_smooth_l1_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for smooth_l1_loss in dygraph."},
    {"linear_interp_v2",
     (PyCFunction)(void (*)(void))eager_api_linear_interp_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for linear_interp_v2 in dygraph."},
    {"momentum", (PyCFunction)(void (*)(void))eager_api_momentum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for momentum in dygraph."},
    {"temporal_shift", (PyCFunction)(void (*)(void))eager_api_temporal_shift,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for temporal_shift in dygraph."},
    {"nce", (PyCFunction)(void (*)(void))eager_api_nce,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for nce in dygraph."},
    {"mv", (PyCFunction)(void (*)(void))eager_api_mv,
     METH_VARARGS | METH_KEYWORDS, "C++ interface function for mv in dygraph."},
    {"proximal_gd", (PyCFunction)(void (*)(void))eager_api_proximal_gd,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for proximal_gd in dygraph."},
    {"memcpy_h2d", (PyCFunction)(void (*)(void))eager_api_memcpy_h2d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for memcpy_h2d in dygraph."},
    {"add_position_encoding",
     (PyCFunction)(void (*)(void))eager_api_add_position_encoding,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for add_position_encoding in dygraph."},
    {"cosh", (PyCFunction)(void (*)(void))eager_api_cosh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cosh in dygraph."},
    {"hash", (PyCFunction)(void (*)(void))eager_api_hash,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for hash in dygraph."},
    {"grad_add", (PyCFunction)(void (*)(void))eager_api_grad_add,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for grad_add in dygraph."},
    {"sign", (PyCFunction)(void (*)(void))eager_api_sign,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sign in dygraph."},
    {"prelu", (PyCFunction)(void (*)(void))eager_api_prelu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for prelu in dygraph."},
    {"linspace", (PyCFunction)(void (*)(void))eager_api_linspace,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for linspace in dygraph."},
    {"fill_diagonal", (PyCFunction)(void (*)(void))eager_api_fill_diagonal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fill_diagonal in dygraph."},
    {"logsigmoid", (PyCFunction)(void (*)(void))eager_api_logsigmoid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logsigmoid in dygraph."},
    {"load_combine", (PyCFunction)(void (*)(void))eager_api_load_combine,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for load_combine in dygraph."},
    {"fetch_v2", (PyCFunction)(void (*)(void))eager_api_fetch_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fetch_v2 in dygraph."},
    {"randperm", (PyCFunction)(void (*)(void))eager_api_randperm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for randperm in dygraph."},
    {"sequence_scatter",
     (PyCFunction)(void (*)(void))eager_api_sequence_scatter,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_scatter in dygraph."},
    {"partial_sum", (PyCFunction)(void (*)(void))eager_api_partial_sum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for partial_sum in dygraph."},
    {"relu6", (PyCFunction)(void (*)(void))eager_api_relu6,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for relu6 in dygraph."},
    {"conv3d", (PyCFunction)(void (*)(void))eager_api_conv3d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv3d in dygraph."},
    {"lstm_unit", (PyCFunction)(void (*)(void))eager_api_lstm_unit,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lstm_unit in dygraph."},
    {"not_equal", (PyCFunction)(void (*)(void))eager_api_not_equal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for not_equal in dygraph."},
    {"transpose2", (PyCFunction)(void (*)(void))eager_api_transpose2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for transpose2 in dygraph."},
    {"uniform_random_batch_size_like",
     (PyCFunction)(void (*)(void))eager_api_uniform_random_batch_size_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for uniform_random_batch_size_like in dygraph."},
    {"unfold", (PyCFunction)(void (*)(void))eager_api_unfold,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unfold in dygraph."},
    {"lrn", (PyCFunction)(void (*)(void))eager_api_lrn,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lrn in dygraph."},
    {"softmax_with_cross_entropy",
     (PyCFunction)(void (*)(void))eager_api_softmax_with_cross_entropy,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for softmax_with_cross_entropy in dygraph."},
    {"isfinite_v2", (PyCFunction)(void (*)(void))eager_api_isfinite_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for isfinite_v2 in dygraph."},
    {"bernoulli", (PyCFunction)(void (*)(void))eager_api_bernoulli,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bernoulli in dygraph."},
    {"max_pool3d_with_index",
     (PyCFunction)(void (*)(void))eager_api_max_pool3d_with_index,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for max_pool3d_with_index in dygraph."},
    {"gaussian_random", (PyCFunction)(void (*)(void))eager_api_gaussian_random,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gaussian_random in dygraph."},
    {"flatten2", (PyCFunction)(void (*)(void))eager_api_flatten2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for flatten2 in dygraph."},
    {"matmul", (PyCFunction)(void (*)(void))eager_api_matmul,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matmul in dygraph."},
    {"cvm", (PyCFunction)(void (*)(void))eager_api_cvm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cvm in dygraph."},
    {"adamax", (PyCFunction)(void (*)(void))eager_api_adamax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for adamax in dygraph."},
    {"masked_select", (PyCFunction)(void (*)(void))eager_api_masked_select,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for masked_select in dygraph."},
    {"range", (PyCFunction)(void (*)(void))eager_api_range,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for range in dygraph."},
    {"bitwise_not", (PyCFunction)(void (*)(void))eager_api_bitwise_not,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bitwise_not in dygraph."},
    {"trace", (PyCFunction)(void (*)(void))eager_api_trace,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for trace in dygraph."},
    {"multinomial", (PyCFunction)(void (*)(void))eager_api_multinomial,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multinomial in dygraph."},
    {"modified_huber_loss",
     (PyCFunction)(void (*)(void))eager_api_modified_huber_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for modified_huber_loss in dygraph."},
    {"roll", (PyCFunction)(void (*)(void))eager_api_roll,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for roll in dygraph."},
    {"squared_l2_distance",
     (PyCFunction)(void (*)(void))eager_api_squared_l2_distance,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for squared_l2_distance in dygraph."},
    {"conv3d_transpose",
     (PyCFunction)(void (*)(void))eager_api_conv3d_transpose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv3d_transpose in dygraph."},
    {"share_data", (PyCFunction)(void (*)(void))eager_api_share_data,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for share_data in dygraph."},
    {"fake_quantize_abs_max",
     (PyCFunction)(void (*)(void))eager_api_fake_quantize_abs_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fake_quantize_abs_max in dygraph."},
    {"unique_with_counts",
     (PyCFunction)(void (*)(void))eager_api_unique_with_counts,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unique_with_counts in dygraph."},
    {"fill", (PyCFunction)(void (*)(void))eager_api_fill,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fill in dygraph."},
    {"concat", (PyCFunction)(void (*)(void))eager_api_concat,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for concat in dygraph."},
    {"fill_zeros_like", (PyCFunction)(void (*)(void))eager_api_fill_zeros_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fill_zeros_like in dygraph."},
    {"hierarchical_sigmoid",
     (PyCFunction)(void (*)(void))eager_api_hierarchical_sigmoid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for hierarchical_sigmoid in dygraph."},
    {"isinf_v2", (PyCFunction)(void (*)(void))eager_api_isinf_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for isinf_v2 in dygraph."},
    {"squeeze", (PyCFunction)(void (*)(void))eager_api_squeeze,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for squeeze in dygraph."},
    {"multiclass_nms2", (PyCFunction)(void (*)(void))eager_api_multiclass_nms2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multiclass_nms2 in dygraph."},
    {"bpr_loss", (PyCFunction)(void (*)(void))eager_api_bpr_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bpr_loss in dygraph."},
    {"fft_c2c", (PyCFunction)(void (*)(void))eager_api_fft_c2c,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fft_c2c in dygraph."},
    {"bicubic_interp_v2",
     (PyCFunction)(void (*)(void))eager_api_bicubic_interp_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bicubic_interp_v2 in dygraph."},
    {"reshape", (PyCFunction)(void (*)(void))eager_api_reshape,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reshape in dygraph."},
    {"coalesce_tensor", (PyCFunction)(void (*)(void))eager_api_coalesce_tensor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for coalesce_tensor in dygraph."},
    {"roi_align", (PyCFunction)(void (*)(void))eager_api_roi_align,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for roi_align in dygraph."},
    {"reshape2", (PyCFunction)(void (*)(void))eager_api_reshape2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reshape2 in dygraph."},
    {"reduce_any", (PyCFunction)(void (*)(void))eager_api_reduce_any,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reduce_any in dygraph."},
    {"unstack", (PyCFunction)(void (*)(void))eager_api_unstack,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unstack in dygraph."},
    {"scatter_nd_add", (PyCFunction)(void (*)(void))eager_api_scatter_nd_add,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for scatter_nd_add in dygraph."},
    {"sequence_reshape",
     (PyCFunction)(void (*)(void))eager_api_sequence_reshape,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_reshape in dygraph."},
    {"bilateral_slice", (PyCFunction)(void (*)(void))eager_api_bilateral_slice,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bilateral_slice in dygraph."},
    {"fill_any_like", (PyCFunction)(void (*)(void))eager_api_fill_any_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fill_any_like in dygraph."},
    {"empty", (PyCFunction)(void (*)(void))eager_api_empty,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for empty in dygraph."},
    {"pad_constant_like",
     (PyCFunction)(void (*)(void))eager_api_pad_constant_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pad_constant_like in dygraph."},
    {"pool2d", (PyCFunction)(void (*)(void))eager_api_pool2d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pool2d in dygraph."},
    {"size", (PyCFunction)(void (*)(void))eager_api_size,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for size in dygraph."},
    {"imag", (PyCFunction)(void (*)(void))eager_api_imag,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for imag in dygraph."},
    {"eigh", (PyCFunction)(void (*)(void))eager_api_eigh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for eigh in dygraph."},
    {"stack", (PyCFunction)(void (*)(void))eager_api_stack,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for stack in dygraph."},
    {"dgc_momentum", (PyCFunction)(void (*)(void))eager_api_dgc_momentum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dgc_momentum in dygraph."},
    {"lamb", (PyCFunction)(void (*)(void))eager_api_lamb,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lamb in dygraph."},
    {"generate_proposals_v2",
     (PyCFunction)(void (*)(void))eager_api_generate_proposals_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for generate_proposals_v2 in dygraph."},
    {"bitwise_or", (PyCFunction)(void (*)(void))eager_api_bitwise_or,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bitwise_or in dygraph."},
    {"gru_unit", (PyCFunction)(void (*)(void))eager_api_gru_unit,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gru_unit in dygraph."},
    {"fake_channel_wise_quantize_dequantize_abs_max",
     (PyCFunction)(
         void (*)(void))eager_api_fake_channel_wise_quantize_dequantize_abs_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fake_channel_wise_quantize_dequantize_abs_max "
     "in dygraph."},
    {"sampling_id", (PyCFunction)(void (*)(void))eager_api_sampling_id,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sampling_id in dygraph."},
    {"unsqueeze2", (PyCFunction)(void (*)(void))eager_api_unsqueeze2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unsqueeze2 in dygraph."},
    {"average_accumulates",
     (PyCFunction)(void (*)(void))eager_api_average_accumulates,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for average_accumulates in dygraph."},
    {"sequence_enumerate",
     (PyCFunction)(void (*)(void))eager_api_sequence_enumerate,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_enumerate in dygraph."},
    {"fusion_seqconv_eltadd_relu",
     (PyCFunction)(void (*)(void))eager_api_fusion_seqconv_eltadd_relu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fusion_seqconv_eltadd_relu in dygraph."},
    {"bce_loss", (PyCFunction)(void (*)(void))eager_api_bce_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bce_loss in dygraph."},
    {"generate_proposal_labels",
     (PyCFunction)(void (*)(void))eager_api_generate_proposal_labels,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for generate_proposal_labels in dygraph."},
    {"im2sequence", (PyCFunction)(void (*)(void))eager_api_im2sequence,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for im2sequence in dygraph."},
    {"isinf", (PyCFunction)(void (*)(void))eager_api_isinf,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for isinf in dygraph."},
    {"adagrad", (PyCFunction)(void (*)(void))eager_api_adagrad,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for adagrad in dygraph."},
    {"linear_chain_crf",
     (PyCFunction)(void (*)(void))eager_api_linear_chain_crf,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for linear_chain_crf in dygraph."},
    {"retinanet_target_assign",
     (PyCFunction)(void (*)(void))eager_api_retinanet_target_assign,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for retinanet_target_assign in dygraph."},
    {"fusion_group", (PyCFunction)(void (*)(void))eager_api_fusion_group,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fusion_group in dygraph."},
    {"teacher_student_sigmoid_loss",
     (PyCFunction)(void (*)(void))eager_api_teacher_student_sigmoid_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for teacher_student_sigmoid_loss in dygraph."},
    {"random_crop", (PyCFunction)(void (*)(void))eager_api_random_crop,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for random_crop in dygraph."},
    {"lookup_table_v2", (PyCFunction)(void (*)(void))eager_api_lookup_table_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lookup_table_v2 in dygraph."},
    {"detection_map", (PyCFunction)(void (*)(void))eager_api_detection_map,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for detection_map in dygraph."},
    {"l1_norm", (PyCFunction)(void (*)(void))eager_api_l1_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for l1_norm in dygraph."},
    {"sqrt", (PyCFunction)(void (*)(void))eager_api_sqrt,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sqrt in dygraph."},
    {"fused_elemwise_activation",
     (PyCFunction)(void (*)(void))eager_api_fused_elemwise_activation,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_elemwise_activation in dygraph."},
    {"slogdeterminant", (PyCFunction)(void (*)(void))eager_api_slogdeterminant,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for slogdeterminant in dygraph."},
    {"share_buffer", (PyCFunction)(void (*)(void))eager_api_share_buffer,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for share_buffer in dygraph."},
    {"bitwise_and", (PyCFunction)(void (*)(void))eager_api_bitwise_and,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bitwise_and in dygraph."},
    {"diag_embed", (PyCFunction)(void (*)(void))eager_api_diag_embed,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for diag_embed in dygraph."},
    {"unbind", (PyCFunction)(void (*)(void))eager_api_unbind,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unbind in dygraph."},
    {"dropout", (PyCFunction)(void (*)(void))eager_api_dropout,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dropout in dygraph."},
    {"moving_average_abs_max_scale",
     (PyCFunction)(void (*)(void))eager_api_moving_average_abs_max_scale,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for moving_average_abs_max_scale in dygraph."},
    {"beam_search", (PyCFunction)(void (*)(void))eager_api_beam_search,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for beam_search in dygraph."},
    {"log_loss", (PyCFunction)(void (*)(void))eager_api_log_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log_loss in dygraph."},
    {"greater_than", (PyCFunction)(void (*)(void))eager_api_greater_than,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for greater_than in dygraph."},
    {"kron", (PyCFunction)(void (*)(void))eager_api_kron,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for kron in dygraph."},
    {"sigmoid_focal_loss",
     (PyCFunction)(void (*)(void))eager_api_sigmoid_focal_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sigmoid_focal_loss in dygraph."},
    {"rmsprop", (PyCFunction)(void (*)(void))eager_api_rmsprop,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for rmsprop in dygraph."},
    {"conv2d", (PyCFunction)(void (*)(void))eager_api_conv2d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv2d in dygraph."},
    {"uniform_random_inplace",
     (PyCFunction)(void (*)(void))eager_api_uniform_random_inplace,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for uniform_random_inplace in dygraph."},
    {"maxout", (PyCFunction)(void (*)(void))eager_api_maxout,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for maxout in dygraph."},
    {"linear_interp", (PyCFunction)(void (*)(void))eager_api_linear_interp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for linear_interp in dygraph."},
    {"auc", (PyCFunction)(void (*)(void))eager_api_auc,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for auc in dygraph."},
    {"logical_or", (PyCFunction)(void (*)(void))eager_api_logical_or,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logical_or in dygraph."},
    {"batch_norm", (PyCFunction)(void (*)(void))eager_api_batch_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for batch_norm in dygraph."},
    {"elementwise_add", (PyCFunction)(void (*)(void))eager_api_elementwise_add,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_add in dygraph."},
    {"acos", (PyCFunction)(void (*)(void))eager_api_acos,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for acos in dygraph."},
    {"unpool", (PyCFunction)(void (*)(void))eager_api_unpool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unpool in dygraph."},
    {"cumprod", (PyCFunction)(void (*)(void))eager_api_cumprod,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cumprod in dygraph."},
    {"sample_logits", (PyCFunction)(void (*)(void))eager_api_sample_logits,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sample_logits in dygraph."},
    {"pull_box_extended_sparse",
     (PyCFunction)(void (*)(void))eager_api_pull_box_extended_sparse,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pull_box_extended_sparse in dygraph."},
    {"crop_tensor", (PyCFunction)(void (*)(void))eager_api_crop_tensor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for crop_tensor in dygraph."},
    {"fill_constant", (PyCFunction)(void (*)(void))eager_api_fill_constant,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fill_constant in dygraph."},
    {"deformable_conv", (PyCFunction)(void (*)(void))eager_api_deformable_conv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for deformable_conv in dygraph."},
    {"generate_mask_labels",
     (PyCFunction)(void (*)(void))eager_api_generate_mask_labels,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for generate_mask_labels in dygraph."},
    {"locality_aware_nms",
     (PyCFunction)(void (*)(void))eager_api_locality_aware_nms,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for locality_aware_nms in dygraph."},
    {"expand_as", (PyCFunction)(void (*)(void))eager_api_expand_as,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for expand_as in dygraph."},
    {"matrix_power", (PyCFunction)(void (*)(void))eager_api_matrix_power,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matrix_power in dygraph."},
    {"greater_equal", (PyCFunction)(void (*)(void))eager_api_greater_equal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for greater_equal in dygraph."},
    {"generate_proposals",
     (PyCFunction)(void (*)(void))eager_api_generate_proposals,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for generate_proposals in dygraph."},
    {"bilinear_interp", (PyCFunction)(void (*)(void))eager_api_bilinear_interp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bilinear_interp in dygraph."},
    {"sigmoid", (PyCFunction)(void (*)(void))eager_api_sigmoid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sigmoid in dygraph."},
    {"inplace_abn", (PyCFunction)(void (*)(void))eager_api_inplace_abn,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for inplace_abn in dygraph."},
    {"softshrink", (PyCFunction)(void (*)(void))eager_api_softshrink,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for softshrink in dygraph."},
    {"mul", (PyCFunction)(void (*)(void))eager_api_mul,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mul in dygraph."},
    {"data_norm", (PyCFunction)(void (*)(void))eager_api_data_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for data_norm in dygraph."},
    {"get_tensor_from_selected_rows",
     (PyCFunction)(void (*)(void))eager_api_get_tensor_from_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for get_tensor_from_selected_rows in dygraph."},
    {"spp", (PyCFunction)(void (*)(void))eager_api_spp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for spp in dygraph."},
    {"floor", (PyCFunction)(void (*)(void))eager_api_floor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for floor in dygraph."},
    {"gelu", (PyCFunction)(void (*)(void))eager_api_gelu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gelu in dygraph."},
    {"retinanet_detection_output",
     (PyCFunction)(void (*)(void))eager_api_retinanet_detection_output,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for retinanet_detection_output in dygraph."},
    {"minus", (PyCFunction)(void (*)(void))eager_api_minus,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for minus in dygraph."},
    {"push_dense", (PyCFunction)(void (*)(void))eager_api_push_dense,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for push_dense in dygraph."},
    {"silu", (PyCFunction)(void (*)(void))eager_api_silu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for silu in dygraph."},
    {"sequence_erase", (PyCFunction)(void (*)(void))eager_api_sequence_erase,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_erase in dygraph."},
    {"real", (PyCFunction)(void (*)(void))eager_api_real,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for real in dygraph."},
    {"nearest_interp_v2",
     (PyCFunction)(void (*)(void))eager_api_nearest_interp_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for nearest_interp_v2 in dygraph."},
    {"dgc_clip_by_norm",
     (PyCFunction)(void (*)(void))eager_api_dgc_clip_by_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dgc_clip_by_norm in dygraph."},
    {"squeeze2", (PyCFunction)(void (*)(void))eager_api_squeeze2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for squeeze2 in dygraph."},
    {"strided_slice", (PyCFunction)(void (*)(void))eager_api_strided_slice,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for strided_slice in dygraph."},
    {"conj", (PyCFunction)(void (*)(void))eager_api_conj,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conj in dygraph."},
    {"precision_recall",
     (PyCFunction)(void (*)(void))eager_api_precision_recall,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for precision_recall in dygraph."},
    {"save", (PyCFunction)(void (*)(void))eager_api_save,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for save in dygraph."},
    {"fusion_seqexpand_concat_fc",
     (PyCFunction)(void (*)(void))eager_api_fusion_seqexpand_concat_fc,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fusion_seqexpand_concat_fc in dygraph."},
    {"fake_quantize_range_abs_max",
     (PyCFunction)(void (*)(void))eager_api_fake_quantize_range_abs_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fake_quantize_range_abs_max in dygraph."},
    {"depthwise_conv2d_transpose",
     (PyCFunction)(void (*)(void))eager_api_depthwise_conv2d_transpose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for depthwise_conv2d_transpose in dygraph."},
    {"positive_negative_pair",
     (PyCFunction)(void (*)(void))eager_api_positive_negative_pair,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for positive_negative_pair in dygraph."},
    {"square", (PyCFunction)(void (*)(void))eager_api_square,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for square in dygraph."},
    {"var_conv_2d", (PyCFunction)(void (*)(void))eager_api_var_conv_2d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for var_conv_2d in dygraph."},
    {"log1p", (PyCFunction)(void (*)(void))eager_api_log1p,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log1p in dygraph."},
    {"fused_softmax_mask_upper_triangle",
     (PyCFunction)(void (*)(void))eager_api_fused_softmax_mask_upper_triangle,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_softmax_mask_upper_triangle in "
     "dygraph."},
    {"clip_by_norm", (PyCFunction)(void (*)(void))eager_api_clip_by_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for clip_by_norm in dygraph."},
    {"atan2", (PyCFunction)(void (*)(void))eager_api_atan2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for atan2 in dygraph."},
    {"box_decoder_and_assign",
     (PyCFunction)(void (*)(void))eager_api_box_decoder_and_assign,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for box_decoder_and_assign in dygraph."},
    {"fft_r2c", (PyCFunction)(void (*)(void))eager_api_fft_r2c,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fft_r2c in dygraph."},
    {"roi_pool", (PyCFunction)(void (*)(void))eager_api_roi_pool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for roi_pool in dygraph."},
    {"overlap_add", (PyCFunction)(void (*)(void))eager_api_overlap_add,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for overlap_add in dygraph."},
    {"fill_constant_batch_size_like",
     (PyCFunction)(void (*)(void))eager_api_fill_constant_batch_size_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fill_constant_batch_size_like in dygraph."},
    {"fill_any", (PyCFunction)(void (*)(void))eager_api_fill_any,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fill_any in dygraph."},
    {"dequantize_log", (PyCFunction)(void (*)(void))eager_api_dequantize_log,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dequantize_log in dygraph."},
    {"max_pool2d_with_index",
     (PyCFunction)(void (*)(void))eager_api_max_pool2d_with_index,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for max_pool2d_with_index in dygraph."},
    {"pad3d", (PyCFunction)(void (*)(void))eager_api_pad3d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pad3d in dygraph."},
    {"norm", (PyCFunction)(void (*)(void))eager_api_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for norm in dygraph."},
    {"viterbi_decode", (PyCFunction)(void (*)(void))eager_api_viterbi_decode,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for viterbi_decode in dygraph."},
    {"mish", (PyCFunction)(void (*)(void))eager_api_mish,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mish in dygraph."},
    {"box_coder", (PyCFunction)(void (*)(void))eager_api_box_coder,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for box_coder in dygraph."},
    {"flatten", (PyCFunction)(void (*)(void))eager_api_flatten,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for flatten in dygraph."},
    {"elementwise_mod", (PyCFunction)(void (*)(void))eager_api_elementwise_mod,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_mod in dygraph."},
    {"margin_cross_entropy",
     (PyCFunction)(void (*)(void))eager_api_margin_cross_entropy,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for margin_cross_entropy in dygraph."},
    {"pull_sparse", (PyCFunction)(void (*)(void))eager_api_pull_sparse,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pull_sparse in dygraph."},
    {"logical_and", (PyCFunction)(void (*)(void))eager_api_logical_and,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logical_and in dygraph."},
    {"pow", (PyCFunction)(void (*)(void))eager_api_pow,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pow in dygraph."},
    {"stanh", (PyCFunction)(void (*)(void))eager_api_stanh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for stanh in dygraph."},
    {"label_smooth", (PyCFunction)(void (*)(void))eager_api_label_smooth,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for label_smooth in dygraph."},
    {"merged_momentum", (PyCFunction)(void (*)(void))eager_api_merged_momentum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for merged_momentum in dygraph."},
    {"ascend_trigger", (PyCFunction)(void (*)(void))eager_api_ascend_trigger,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for ascend_trigger in dygraph."},
    {"fused_feedforward",
     (PyCFunction)(void (*)(void))eager_api_fused_feedforward,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_feedforward in dygraph."},
    {"rpn_target_assign",
     (PyCFunction)(void (*)(void))eager_api_rpn_target_assign,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for rpn_target_assign in dygraph."},
    {"roi_perspective_transform",
     (PyCFunction)(void (*)(void))eager_api_roi_perspective_transform,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for roi_perspective_transform in dygraph."},
    {"expand", (PyCFunction)(void (*)(void))eager_api_expand,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for expand in dygraph."},
    {"prroi_pool", (PyCFunction)(void (*)(void))eager_api_prroi_pool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for prroi_pool in dygraph."},
    {"pool3d", (PyCFunction)(void (*)(void))eager_api_pool3d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pool3d in dygraph."},
    {"memcpy", (PyCFunction)(void (*)(void))eager_api_memcpy,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for memcpy in dygraph."},
    {"distribute_fpn_proposals",
     (PyCFunction)(void (*)(void))eager_api_distribute_fpn_proposals,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for distribute_fpn_proposals in dygraph."},
    {"frame", (PyCFunction)(void (*)(void))eager_api_frame,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for frame in dygraph."},
    {"bincount", (PyCFunction)(void (*)(void))eager_api_bincount,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bincount in dygraph."},
    {"shape", (PyCFunction)(void (*)(void))eager_api_shape,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for shape in dygraph."},
    {"group_norm", (PyCFunction)(void (*)(void))eager_api_group_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for group_norm in dygraph."},
    {"resnet_unit", (PyCFunction)(void (*)(void))eager_api_resnet_unit,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for resnet_unit in dygraph."},
    {"sequence_expand_as",
     (PyCFunction)(void (*)(void))eager_api_sequence_expand_as,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_expand_as in dygraph."},
    {"cos_sim", (PyCFunction)(void (*)(void))eager_api_cos_sim,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cos_sim in dygraph."},
    {"eigvals", (PyCFunction)(void (*)(void))eager_api_eigvals,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for eigvals in dygraph."},
    {"save_combine", (PyCFunction)(void (*)(void))eager_api_save_combine,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for save_combine in dygraph."},
    {"class_center_sample",
     (PyCFunction)(void (*)(void))eager_api_class_center_sample,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for class_center_sample in dygraph."},
    {"read_file", (PyCFunction)(void (*)(void))eager_api_read_file,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for read_file in dygraph."},
    {"isfinite", (PyCFunction)(void (*)(void))eager_api_isfinite,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for isfinite in dygraph."},
    {"arg_max", (PyCFunction)(void (*)(void))eager_api_arg_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for arg_max in dygraph."},
    {"equal", (PyCFunction)(void (*)(void))eager_api_equal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for equal in dygraph."},
    {"fake_dequantize_max_abs",
     (PyCFunction)(void (*)(void))eager_api_fake_dequantize_max_abs,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fake_dequantize_max_abs in dygraph."},
    {"qr", (PyCFunction)(void (*)(void))eager_api_qr,
     METH_VARARGS | METH_KEYWORDS, "C++ interface function for qr in dygraph."},
    {"anchor_generator",
     (PyCFunction)(void (*)(void))eager_api_anchor_generator,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for anchor_generator in dygraph."},
    {"layer_norm", (PyCFunction)(void (*)(void))eager_api_layer_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for layer_norm in dygraph."},
    {"merge_selected_rows",
     (PyCFunction)(void (*)(void))eager_api_merge_selected_rows,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for merge_selected_rows in dygraph."},
    {"less_equal", (PyCFunction)(void (*)(void))eager_api_less_equal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for less_equal in dygraph."},
    {"rnn", (PyCFunction)(void (*)(void))eager_api_rnn,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for rnn in dygraph."},
    {"fusion_lstm", (PyCFunction)(void (*)(void))eager_api_fusion_lstm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fusion_lstm in dygraph."},
    {"lars_momentum", (PyCFunction)(void (*)(void))eager_api_lars_momentum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lars_momentum in dygraph."},
    {"hard_sigmoid", (PyCFunction)(void (*)(void))eager_api_hard_sigmoid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for hard_sigmoid in dygraph."},
    {"isnan", (PyCFunction)(void (*)(void))eager_api_isnan,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for isnan in dygraph."},
    {"elementwise_floordiv",
     (PyCFunction)(void (*)(void))eager_api_elementwise_floordiv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_floordiv in dygraph."},
    {"correlation", (PyCFunction)(void (*)(void))eager_api_correlation,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for correlation in dygraph."},
    {"histogram", (PyCFunction)(void (*)(void))eager_api_histogram,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for histogram in dygraph."},
    {"gather_tree", (PyCFunction)(void (*)(void))eager_api_gather_tree,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gather_tree in dygraph."},
    {"segment_pool", (PyCFunction)(void (*)(void))eager_api_segment_pool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for segment_pool in dygraph."},
    {"sync_batch_norm", (PyCFunction)(void (*)(void))eager_api_sync_batch_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sync_batch_norm in dygraph."},
    {"fusion_repeated_fc_relu",
     (PyCFunction)(void (*)(void))eager_api_fusion_repeated_fc_relu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fusion_repeated_fc_relu in dygraph."},
    {"nop", (PyCFunction)(void (*)(void))eager_api_nop,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for nop in dygraph."},
    {"fused_attention", (PyCFunction)(void (*)(void))eager_api_fused_attention,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_attention in dygraph."},
    {"expand_as_v2", (PyCFunction)(void (*)(void))eager_api_expand_as_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for expand_as_v2 in dygraph."},
    {"filter_by_instag",
     (PyCFunction)(void (*)(void))eager_api_filter_by_instag,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for filter_by_instag in dygraph."},
    {"diag_v2", (PyCFunction)(void (*)(void))eager_api_diag_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for diag_v2 in dygraph."},
    {"pull_box_sparse", (PyCFunction)(void (*)(void))eager_api_pull_box_sparse,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pull_box_sparse in dygraph."},
    {"nll_loss", (PyCFunction)(void (*)(void))eager_api_nll_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for nll_loss in dygraph."},
    {"dot", (PyCFunction)(void (*)(void))eager_api_dot,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dot in dygraph."},
    {"scale", (PyCFunction)(void (*)(void))eager_api_scale,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for scale in dygraph."},
    {"ncclBcast", (PyCFunction)(void (*)(void))eager_api_ncclBcast,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for ncclBcast in dygraph."},
    {"shuffle_batch", (PyCFunction)(void (*)(void))eager_api_shuffle_batch,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for shuffle_batch in dygraph."},
    {"ncclReduce", (PyCFunction)(void (*)(void))eager_api_ncclReduce,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for ncclReduce in dygraph."},
    {"diag", (PyCFunction)(void (*)(void))eager_api_diag,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for diag in dygraph."},
    {"multiplex", (PyCFunction)(void (*)(void))eager_api_multiplex,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multiplex in dygraph."},
    {"leaky_relu", (PyCFunction)(void (*)(void))eager_api_leaky_relu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for leaky_relu in dygraph."},
    {"allclose", (PyCFunction)(void (*)(void))eager_api_allclose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for allclose in dygraph."},
    {"adamw", (PyCFunction)(void (*)(void))eager_api_adamw,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for adamw in dygraph."},
    {"elementwise_pow", (PyCFunction)(void (*)(void))eager_api_elementwise_pow,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_pow in dygraph."},
    {"prior_box", (PyCFunction)(void (*)(void))eager_api_prior_box,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for prior_box in dygraph."},
    {"p_norm", (PyCFunction)(void (*)(void))eager_api_p_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for p_norm in dygraph."},
    {"unique_consecutive",
     (PyCFunction)(void (*)(void))eager_api_unique_consecutive,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unique_consecutive in dygraph."},
    {"lod_reset", (PyCFunction)(void (*)(void))eager_api_lod_reset,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lod_reset in dygraph."},
    {"pad", (PyCFunction)(void (*)(void))eager_api_pad,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pad in dygraph."},
    {"sequence_conv", (PyCFunction)(void (*)(void))eager_api_sequence_conv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_conv in dygraph."},
    {"log10", (PyCFunction)(void (*)(void))eager_api_log10,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log10 in dygraph."},
    {"set_value", (PyCFunction)(void (*)(void))eager_api_set_value,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for set_value in dygraph."},
    {"bitwise_xor", (PyCFunction)(void (*)(void))eager_api_bitwise_xor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bitwise_xor in dygraph."},
    {"center_loss", (PyCFunction)(void (*)(void))eager_api_center_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for center_loss in dygraph."},
    {"randint", (PyCFunction)(void (*)(void))eager_api_randint,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for randint in dygraph."},
    {"attention_lstm", (PyCFunction)(void (*)(void))eager_api_attention_lstm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for attention_lstm in dygraph."},
    {"uniform_random", (PyCFunction)(void (*)(void))eager_api_uniform_random,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for uniform_random in dygraph."},
    {"slice", (PyCFunction)(void (*)(void))eager_api_slice,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for slice in dygraph."},
    {"meshgrid", (PyCFunction)(void (*)(void))eager_api_meshgrid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for meshgrid in dygraph."},
    {"hard_swish", (PyCFunction)(void (*)(void))eager_api_hard_swish,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for hard_swish in dygraph."},
    {"sin", (PyCFunction)(void (*)(void))eager_api_sin,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sin in dygraph."},
    {"mean_iou", (PyCFunction)(void (*)(void))eager_api_mean_iou,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mean_iou in dygraph."},
    {"pad2d", (PyCFunction)(void (*)(void))eager_api_pad2d,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pad2d in dygraph."},
    {"inverse", (PyCFunction)(void (*)(void))eager_api_inverse,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for inverse in dygraph."},
    {"spectral_norm", (PyCFunction)(void (*)(void))eager_api_spectral_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for spectral_norm in dygraph."},
    {"shuffle_channel", (PyCFunction)(void (*)(void))eager_api_shuffle_channel,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for shuffle_channel in dygraph."},
    {"psroi_pool", (PyCFunction)(void (*)(void))eager_api_psroi_pool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for psroi_pool in dygraph."},
    {"seed", (PyCFunction)(void (*)(void))eager_api_seed,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for seed in dygraph."},
    {"ceil", (PyCFunction)(void (*)(void))eager_api_ceil,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for ceil in dygraph."},
    {"eig", (PyCFunction)(void (*)(void))eager_api_eig,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for eig in dygraph."},
    {"reduce_min", (PyCFunction)(void (*)(void))eager_api_reduce_min,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reduce_min in dygraph."},
    {"cos", (PyCFunction)(void (*)(void))eager_api_cos,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cos in dygraph."},
    {"ncclAllReduce", (PyCFunction)(void (*)(void))eager_api_ncclAllReduce,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for ncclAllReduce in dygraph."},
    {"cudnn_lstm", (PyCFunction)(void (*)(void))eager_api_cudnn_lstm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cudnn_lstm in dygraph."},
    {"reduce_sum", (PyCFunction)(void (*)(void))eager_api_reduce_sum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reduce_sum in dygraph."},
    {"digamma", (PyCFunction)(void (*)(void))eager_api_digamma,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for digamma in dygraph."},
    {"assign_value", (PyCFunction)(void (*)(void))eager_api_assign_value,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for assign_value in dygraph."},
    {"increment", (PyCFunction)(void (*)(void))eager_api_increment,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for increment in dygraph."},
    {"tdm_sampler", (PyCFunction)(void (*)(void))eager_api_tdm_sampler,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tdm_sampler in dygraph."},
    {"fused_softmax_mask",
     (PyCFunction)(void (*)(void))eager_api_fused_softmax_mask,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_softmax_mask in dygraph."},
    {"sequence_reverse",
     (PyCFunction)(void (*)(void))eager_api_sequence_reverse,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_reverse in dygraph."},
    {"eigvalsh", (PyCFunction)(void (*)(void))eager_api_eigvalsh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for eigvalsh in dygraph."},
    {"diagonal", (PyCFunction)(void (*)(void))eager_api_diagonal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for diagonal in dygraph."},
    {"trunc", (PyCFunction)(void (*)(void))eager_api_trunc,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for trunc in dygraph."},
    {"log2", (PyCFunction)(void (*)(void))eager_api_log2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log2 in dygraph."},
    {"marker", (PyCFunction)(void (*)(void))eager_api_marker,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for marker in dygraph."},
    {"tanh", (PyCFunction)(void (*)(void))eager_api_tanh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tanh in dygraph."},
    {"yolov3_loss", (PyCFunction)(void (*)(void))eager_api_yolov3_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for yolov3_loss in dygraph."},
    {"graph_send_recv", (PyCFunction)(void (*)(void))eager_api_graph_send_recv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for graph_send_recv in dygraph."},
    {"accuracy", (PyCFunction)(void (*)(void))eager_api_accuracy,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for accuracy in dygraph."},
    {"atan", (PyCFunction)(void (*)(void))eager_api_atan,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for atan in dygraph."},
    {"less_than", (PyCFunction)(void (*)(void))eager_api_less_than,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for less_than in dygraph."},
    {"unsqueeze", (PyCFunction)(void (*)(void))eager_api_unsqueeze,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for unsqueeze in dygraph."},
    {"crf_decoding", (PyCFunction)(void (*)(void))eager_api_crf_decoding,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for crf_decoding in dygraph."},
    {"log_softmax", (PyCFunction)(void (*)(void))eager_api_log_softmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for log_softmax in dygraph."},
    {"ftrl", (PyCFunction)(void (*)(void))eager_api_ftrl,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for ftrl in dygraph."},
    {"matrix_nms", (PyCFunction)(void (*)(void))eager_api_matrix_nms,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for matrix_nms in dygraph."},
    {"top_k_v2", (PyCFunction)(void (*)(void))eager_api_top_k_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for top_k_v2 in dygraph."},
    {"cast", (PyCFunction)(void (*)(void))eager_api_cast,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cast in dygraph."},
    {"tanh_shrink", (PyCFunction)(void (*)(void))eager_api_tanh_shrink,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tanh_shrink in dygraph."},
    {"hard_shrink", (PyCFunction)(void (*)(void))eager_api_hard_shrink,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for hard_shrink in dygraph."},
    {"multiclass_nms", (PyCFunction)(void (*)(void))eager_api_multiclass_nms,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multiclass_nms in dygraph."},
    {"fusion_transpose_flatten_concat",
     (PyCFunction)(void (*)(void))eager_api_fusion_transpose_flatten_concat,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fusion_transpose_flatten_concat in dygraph."},
    {"sequence_unpad", (PyCFunction)(void (*)(void))eager_api_sequence_unpad,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_unpad in dygraph."},
    {"fused_elemwise_add_activation",
     (PyCFunction)(void (*)(void))eager_api_fused_elemwise_add_activation,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_elemwise_add_activation in dygraph."},
    {"pull_sparse_v2", (PyCFunction)(void (*)(void))eager_api_pull_sparse_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pull_sparse_v2 in dygraph."},
    {"frobenius_norm", (PyCFunction)(void (*)(void))eager_api_frobenius_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for frobenius_norm in dygraph."},
    {"crop", (PyCFunction)(void (*)(void))eager_api_crop,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for crop in dygraph."},
    {"cross_entropy2", (PyCFunction)(void (*)(void))eager_api_cross_entropy2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cross_entropy2 in dygraph."},
    {"skip_layernorm", (PyCFunction)(void (*)(void))eager_api_skip_layernorm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for skip_layernorm in dygraph."},
    {"tdm_child", (PyCFunction)(void (*)(void))eager_api_tdm_child,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tdm_child in dygraph."},
    {"fused_embedding_seq_pool",
     (PyCFunction)(void (*)(void))eager_api_fused_embedding_seq_pool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_embedding_seq_pool in dygraph."},
    {"erf", (PyCFunction)(void (*)(void))eager_api_erf,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for erf in dygraph."},
    {"conv2d_inception_fusion",
     (PyCFunction)(void (*)(void))eager_api_conv2d_inception_fusion,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv2d_inception_fusion in dygraph."},
    {"trilinear_interp",
     (PyCFunction)(void (*)(void))eager_api_trilinear_interp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for trilinear_interp in dygraph."},
    {"logsumexp", (PyCFunction)(void (*)(void))eager_api_logsumexp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logsumexp in dygraph."},
    {"fusion_seqpool_concat",
     (PyCFunction)(void (*)(void))eager_api_fusion_seqpool_concat,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fusion_seqpool_concat in dygraph."},
    {"alloc_float_status",
     (PyCFunction)(void (*)(void))eager_api_alloc_float_status,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for alloc_float_status in dygraph."},
    {"sequence_concat", (PyCFunction)(void (*)(void))eager_api_sequence_concat,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_concat in dygraph."},
    {"fusion_seqpool_cvm_concat",
     (PyCFunction)(void (*)(void))eager_api_fusion_seqpool_cvm_concat,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fusion_seqpool_cvm_concat in dygraph."},
    {"similarity_focus",
     (PyCFunction)(void (*)(void))eager_api_similarity_focus,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for similarity_focus in dygraph."},
    {"argsort", (PyCFunction)(void (*)(void))eager_api_argsort,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for argsort in dygraph."},
    {"sequence_expand", (PyCFunction)(void (*)(void))eager_api_sequence_expand,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_expand in dygraph."},
    {"sgd", (PyCFunction)(void (*)(void))eager_api_sgd,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sgd in dygraph."},
    {"fused_bn_add_activation",
     (PyCFunction)(void (*)(void))eager_api_fused_bn_add_activation,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_bn_add_activation in dygraph."},
    {"bilinear_interp_v2",
     (PyCFunction)(void (*)(void))eager_api_bilinear_interp_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bilinear_interp_v2 in dygraph."},
    {"clip", (PyCFunction)(void (*)(void))eager_api_clip,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for clip in dygraph."},
    {"deformable_conv_v1",
     (PyCFunction)(void (*)(void))eager_api_deformable_conv_v1,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for deformable_conv_v1 in dygraph."},
    {"hinge_loss", (PyCFunction)(void (*)(void))eager_api_hinge_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for hinge_loss in dygraph."},
    {"determinant", (PyCFunction)(void (*)(void))eager_api_determinant,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for determinant in dygraph."},
    {"conv2d_transpose",
     (PyCFunction)(void (*)(void))eager_api_conv2d_transpose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv2d_transpose in dygraph."},
    {"memcpy_d2h", (PyCFunction)(void (*)(void))eager_api_memcpy_d2h,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for memcpy_d2h in dygraph."},
    {"softsign", (PyCFunction)(void (*)(void))eager_api_softsign,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for softsign in dygraph."},
    {"fake_quantize_dequantize_abs_max",
     (PyCFunction)(void (*)(void))eager_api_fake_quantize_dequantize_abs_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fake_quantize_dequantize_abs_max in dygraph."},
    {"broadcast_tensors",
     (PyCFunction)(void (*)(void))eager_api_broadcast_tensors,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for broadcast_tensors in dygraph."},
    {"grid_sampler", (PyCFunction)(void (*)(void))eager_api_grid_sampler,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for grid_sampler in dygraph."},
    {"fft_c2r", (PyCFunction)(void (*)(void))eager_api_fft_c2r,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fft_c2r in dygraph."},
    {"pyramid_hash", (PyCFunction)(void (*)(void))eager_api_pyramid_hash,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pyramid_hash in dygraph."},
    {"fake_quantize_dequantize_moving_average_abs_max",
     (PyCFunction)(void (*)(
         void))eager_api_fake_quantize_dequantize_moving_average_abs_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for "
     "fake_quantize_dequantize_moving_average_abs_max in dygraph."},
    {"multi_dot", (PyCFunction)(void (*)(void))eager_api_multi_dot,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for multi_dot in dygraph."},
    {"sequence_pool", (PyCFunction)(void (*)(void))eager_api_sequence_pool,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_pool in dygraph."},
    {"transpose", (PyCFunction)(void (*)(void))eager_api_transpose,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for transpose in dygraph."},
    {"top_k", (PyCFunction)(void (*)(void))eager_api_top_k,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for top_k in dygraph."},
    {"dist", (PyCFunction)(void (*)(void))eager_api_dist,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dist in dygraph."},
    {"affine_grid", (PyCFunction)(void (*)(void))eager_api_affine_grid,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for affine_grid in dygraph."},
    {"gaussian_random_batch_size_like",
     (PyCFunction)(void (*)(void))eager_api_gaussian_random_batch_size_like,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gaussian_random_batch_size_like in dygraph."},
    {"fake_channel_wise_dequantize_max_abs",
     (PyCFunction)(
         void (*)(void))eager_api_fake_channel_wise_dequantize_max_abs,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fake_channel_wise_dequantize_max_abs in "
     "dygraph."},
    {"reciprocal", (PyCFunction)(void (*)(void))eager_api_reciprocal,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reciprocal in dygraph."},
    {"sequence_mask", (PyCFunction)(void (*)(void))eager_api_sequence_mask,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_mask in dygraph."},
    {"fill_diagonal_tensor",
     (PyCFunction)(void (*)(void))eager_api_fill_diagonal_tensor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fill_diagonal_tensor in dygraph."},
    {"abs", (PyCFunction)(void (*)(void))eager_api_abs,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for abs in dygraph."},
    {"partial_concat", (PyCFunction)(void (*)(void))eager_api_partial_concat,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for partial_concat in dygraph."},
    {"elu", (PyCFunction)(void (*)(void))eager_api_elu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elu in dygraph."},
    {"index_select", (PyCFunction)(void (*)(void))eager_api_index_select,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for index_select in dygraph."},
    {"row_conv", (PyCFunction)(void (*)(void))eager_api_row_conv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for row_conv in dygraph."},
    {"cross", (PyCFunction)(void (*)(void))eager_api_cross,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cross in dygraph."},
    {"elementwise_mul", (PyCFunction)(void (*)(void))eager_api_elementwise_mul,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_mul in dygraph."},
    {"decayed_adagrad", (PyCFunction)(void (*)(void))eager_api_decayed_adagrad,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for decayed_adagrad in dygraph."},
    {"bipartite_match", (PyCFunction)(void (*)(void))eager_api_bipartite_match,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bipartite_match in dygraph."},
    {"run_program", (PyCFunction)(void (*)(void))eager_api_run_program,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for run_program in dygraph."},
    {"fake_quantize_moving_average_abs_max",
     (PyCFunction)(
         void (*)(void))eager_api_fake_quantize_moving_average_abs_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fake_quantize_moving_average_abs_max in "
     "dygraph."},
    {"mine_hard_examples",
     (PyCFunction)(void (*)(void))eager_api_mine_hard_examples,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mine_hard_examples in dygraph."},
    {"target_assign", (PyCFunction)(void (*)(void))eager_api_target_assign,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for target_assign in dygraph."},
    {"lstm", (PyCFunction)(void (*)(void))eager_api_lstm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lstm in dygraph."},
    {"truncated_gaussian_random",
     (PyCFunction)(void (*)(void))eager_api_truncated_gaussian_random,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for truncated_gaussian_random in dygraph."},
    {"match_matrix_tensor",
     (PyCFunction)(void (*)(void))eager_api_match_matrix_tensor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for match_matrix_tensor in dygraph."},
    {"elementwise_div", (PyCFunction)(void (*)(void))eager_api_elementwise_div,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_div in dygraph."},
    {"kldiv_loss", (PyCFunction)(void (*)(void))eager_api_kldiv_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for kldiv_loss in dygraph."},
    {"cumsum", (PyCFunction)(void (*)(void))eager_api_cumsum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cumsum in dygraph."},
    {"sum", (PyCFunction)(void (*)(void))eager_api_sum,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sum in dygraph."},
    {"proximal_adagrad",
     (PyCFunction)(void (*)(void))eager_api_proximal_adagrad,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for proximal_adagrad in dygraph."},
    {"update_loss_scaling",
     (PyCFunction)(void (*)(void))eager_api_update_loss_scaling,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for update_loss_scaling in dygraph."},
    {"shard_index", (PyCFunction)(void (*)(void))eager_api_shard_index,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for shard_index in dygraph."},
    {"selu", (PyCFunction)(void (*)(void))eager_api_selu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for selu in dygraph."},
    {"mean", (PyCFunction)(void (*)(void))eager_api_mean,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for mean in dygraph."},
    {"gumbel_softmax", (PyCFunction)(void (*)(void))eager_api_gumbel_softmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gumbel_softmax in dygraph."},
    {"sequence_pad", (PyCFunction)(void (*)(void))eager_api_sequence_pad,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sequence_pad in dygraph."},
    {"tree_conv", (PyCFunction)(void (*)(void))eager_api_tree_conv,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tree_conv in dygraph."},
    {"assign", (PyCFunction)(void (*)(void))eager_api_assign,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for assign in dygraph."},
    {"flatten_contiguous_range",
     (PyCFunction)(void (*)(void))eager_api_flatten_contiguous_range,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for flatten_contiguous_range in dygraph."},
    {"tril_triu", (PyCFunction)(void (*)(void))eager_api_tril_triu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for tril_triu in dygraph."},
    {"brelu", (PyCFunction)(void (*)(void))eager_api_brelu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for brelu in dygraph."},
    {"celu", (PyCFunction)(void (*)(void))eager_api_celu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for celu in dygraph."},
    {"reduce_mean", (PyCFunction)(void (*)(void))eager_api_reduce_mean,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reduce_mean in dygraph."},
    {"sinh", (PyCFunction)(void (*)(void))eager_api_sinh,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for sinh in dygraph."},
    {"rank_loss", (PyCFunction)(void (*)(void))eager_api_rank_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for rank_loss in dygraph."},
    {"reduce_max", (PyCFunction)(void (*)(void))eager_api_reduce_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reduce_max in dygraph."},
    {"fusion_gru", (PyCFunction)(void (*)(void))eager_api_fusion_gru,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fusion_gru in dygraph."},
    {"fill_zeros_like2",
     (PyCFunction)(void (*)(void))eager_api_fill_zeros_like2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fill_zeros_like2 in dygraph."},
    {"expm1", (PyCFunction)(void (*)(void))eager_api_expm1,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for expm1 in dygraph."},
    {"squared_l2_norm", (PyCFunction)(void (*)(void))eager_api_squared_l2_norm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for squared_l2_norm in dygraph."},
    {"elementwise_sub", (PyCFunction)(void (*)(void))eager_api_elementwise_sub,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_sub in dygraph."},
    {"margin_rank_loss",
     (PyCFunction)(void (*)(void))eager_api_margin_rank_loss,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for margin_rank_loss in dygraph."},
    {"faster_tokenizer",
     (PyCFunction)(void (*)(void))eager_api_faster_tokenizer,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for faster_tokenizer in dygraph."},
    {"relu", (PyCFunction)(void (*)(void))eager_api_relu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for relu in dygraph."},
    {"is_empty", (PyCFunction)(void (*)(void))eager_api_is_empty,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for is_empty in dygraph."},
    {"reduce_all", (PyCFunction)(void (*)(void))eager_api_reduce_all,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reduce_all in dygraph."},
    {"edit_distance", (PyCFunction)(void (*)(void))eager_api_edit_distance,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for edit_distance in dygraph."},
    {"bmm", (PyCFunction)(void (*)(void))eager_api_bmm,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for bmm in dygraph."},
    {"yolo_box", (PyCFunction)(void (*)(void))eager_api_yolo_box,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for yolo_box in dygraph."},
    {"soft_relu", (PyCFunction)(void (*)(void))eager_api_soft_relu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for soft_relu in dygraph."},
    {"density_prior_box",
     (PyCFunction)(void (*)(void))eager_api_density_prior_box,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for density_prior_box in dygraph."},
    {"eye", (PyCFunction)(void (*)(void))eager_api_eye,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for eye in dygraph."},
    {"swish", (PyCFunction)(void (*)(void))eager_api_swish,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for swish in dygraph."},
    {"cross_entropy", (PyCFunction)(void (*)(void))eager_api_cross_entropy,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cross_entropy in dygraph."},
    {"dpsgd", (PyCFunction)(void (*)(void))eager_api_dpsgd,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dpsgd in dygraph."},
    {"cholesky", (PyCFunction)(void (*)(void))eager_api_cholesky,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for cholesky in dygraph."},
    {"batch_fc", (PyCFunction)(void (*)(void))eager_api_batch_fc,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for batch_fc in dygraph."},
    {"nearest_interp", (PyCFunction)(void (*)(void))eager_api_nearest_interp,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for nearest_interp in dygraph."},
    {"gather", (PyCFunction)(void (*)(void))eager_api_gather,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for gather in dygraph."},
    {"trilinear_interp_v2",
     (PyCFunction)(void (*)(void))eager_api_trilinear_interp_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for trilinear_interp_v2 in dygraph."},
    {"box_clip", (PyCFunction)(void (*)(void))eager_api_box_clip,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for box_clip in dygraph."},
    {"isnan_v2", (PyCFunction)(void (*)(void))eager_api_isnan_v2,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for isnan_v2 in dygraph."},
    {"softmax", (PyCFunction)(void (*)(void))eager_api_softmax,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for softmax in dygraph."},
    {"conv2d_fusion", (PyCFunction)(void (*)(void))eager_api_conv2d_fusion,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for conv2d_fusion in dygraph."},
    {"fused_batch_norm_act",
     (PyCFunction)(void (*)(void))eager_api_fused_batch_norm_act,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fused_batch_norm_act in dygraph."},
    {"get_float_status",
     (PyCFunction)(void (*)(void))eager_api_get_float_status,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for get_float_status in dygraph."},
    {"index_sample", (PyCFunction)(void (*)(void))eager_api_index_sample,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for index_sample in dygraph."},
    {"elementwise_min", (PyCFunction)(void (*)(void))eager_api_elementwise_min,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for elementwise_min in dygraph."},
    {"logical_not", (PyCFunction)(void (*)(void))eager_api_logical_not,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for logical_not in dygraph."},
    {"collect_fpn_proposals",
     (PyCFunction)(void (*)(void))eager_api_collect_fpn_proposals,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for collect_fpn_proposals in dygraph."},
    {"pixel_shuffle", (PyCFunction)(void (*)(void))eager_api_pixel_shuffle,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for pixel_shuffle in dygraph."},
    {"thresholded_relu",
     (PyCFunction)(void (*)(void))eager_api_thresholded_relu,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for thresholded_relu in dygraph."},
    {"polygon_box_transform",
     (PyCFunction)(void (*)(void))eager_api_polygon_box_transform,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for polygon_box_transform in dygraph."},
    {"lookup_table_dequant",
     (PyCFunction)(void (*)(void))eager_api_lookup_table_dequant,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for lookup_table_dequant in dygraph."},
    {"warpctc", (PyCFunction)(void (*)(void))eager_api_warpctc,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for warpctc in dygraph."},
    {"fake_channel_wise_quantize_abs_max",
     (PyCFunction)(void (*)(void))eager_api_fake_channel_wise_quantize_abs_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for fake_channel_wise_quantize_abs_max in "
     "dygraph."},
    {"dequantize_abs_max",
     (PyCFunction)(void (*)(void))eager_api_dequantize_abs_max,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dequantize_abs_max in dygraph."},
    {"svd", (PyCFunction)(void (*)(void))eager_api_svd,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for svd in dygraph."},
    {"flip", (PyCFunction)(void (*)(void))eager_api_flip,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for flip in dygraph."},
    {nullptr, nullptr, 0, nullptr}};

inline void BindEagerOpFunctions(pybind11::module *module) {
  auto m = module->def_submodule("ops");
  if (PyModule_AddFunctions(m.ptr(), ExtestMethods) < 0) {
    PADDLE_THROW(
        platform::errors::Fatal("Add functions to core.eager.ops failed!"));
  }

  InitOpsAttrTypeMap();
}

}  // namespace pybind
}  // namespace paddle
