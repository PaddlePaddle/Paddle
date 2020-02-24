/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <xxhash.h>
#include <algorithm>
#include <cmath>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/search_compute.h"

extern "C" {
#include "math/bloomfilter.h"
}

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

class PyramidHashOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X (Tensor, MUST be Tensor<!!!_int32_!!!>) Input variable which "
             "should contain lod information.");
    AddInput("W", "W (Tensor)");
    AddInput("WhiteList", "WhiteList (Tensor)");
    AddInput("BlackList", "BlackList (Tensor)");
    AddAttr<int>("num_emb", "num_emb").SetDefault(0).EqualGreaterThan(0);
    AddAttr<int>("space_len", "space_len").SetDefault(0).EqualGreaterThan(0);
    AddAttr<int>("pyramid_layer", "pyramid_layer (must be >= 2)")
        .SetDefault(2)
        .EqualGreaterThan(2);
    AddAttr<int>("rand_len", "rand_len").SetDefault(0).EqualGreaterThan(0);
    AddAttr<float>("drop_out_percent", "drop_out_percent")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddAttr<int>("is_training", "is_training")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddAttr<bool>("use_filter", "use_filter").SetDefault(true);
    AddAttr<int>("white_list_len", "white_list_len")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddAttr<int>("black_list_len", "black_list_len")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddAttr<int>("seed", "seed").SetDefault(0).EqualGreaterThan(0);
    AddAttr<float>("lr", "learning rate").SetDefault(0.0).EqualGreaterThan(0.0);
    AddAttr<std::string>(
        "distribute_update_vars",
        "['PyramidHash_emb_0','Filter']"
        "Decided which params should be updated in distribute training. "
        "Used in Distribute Transpiler to create a trainer/server program.")
        .SetDefault("");
    AddOutput("Out", "Out (Tensor, default Tensor<float>) Output variable");
    AddOutput("DropPos", "Out (Tensor, Tensor<int>) Output variable");
    AddOutput("X_Temp_Out", "Out (Tensor, Tensor<int>) Output variable")
        .AsIntermediate();

    AddComment(R"DOC(
      PyramidHash

      NOTE: only support 'float32' data type now.

    )DOC");
  }
};

class PyramidHashOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true, "X(Input) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true, "W(Input) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Out(Output) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("DropPos"), true,
                      "DropPos(TMP Output) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "The rank of X(Input) should be 2.");

    auto w_dims = ctx->GetInputDim("W");
    PADDLE_ENFORCE_EQ(w_dims.size(), 2, "W should be 2-D tensor");

    int space_len = ctx->Attrs().Get<int>("space_len");
    int rand_len = ctx->Attrs().Get<int>("rand_len");

    PADDLE_ENFORCE_EQ(w_dims[0], space_len + rand_len,
                      "w_dims[0] should be equal to (space_len + rand_len)");
    PADDLE_ENFORCE_EQ(w_dims[1], 1, "w_dims[1] should be equal to 1");

    int num_emb = ctx->Attrs().Get<int>("num_emb");
    PADDLE_ENFORCE_EQ(num_emb % rand_len, 0,
                      "random length should mod embedding size");

    int white_list_len = ctx->Attrs().Get<int>("white_list_len");
    if (white_list_len > 0) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("WhiteList"), true,
          "WhiteList(Input) should not be null when white_list_len > 0");
      auto wl_dims = ctx->GetInputDim("WhiteList");
      PADDLE_ENFORCE_EQ(wl_dims.size(), 2, "WhiteList should be 2-D tensor");
      PADDLE_ENFORCE_EQ(wl_dims[0], white_list_len,
                        "wl_dims[0] should be equal to white_list_len");
      PADDLE_ENFORCE_EQ(wl_dims[1], 1, "wl_dims[1] should be equal to 1");
    }

    int black_list_len = ctx->Attrs().Get<int>("black_list_len");
    if (black_list_len > 0) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("BlackList"), true,
          "BlackList(Input) should not be null when black_list_len > 0");
      auto bl_dims = ctx->GetInputDim("BlackList");
      PADDLE_ENFORCE_EQ(bl_dims.size(), 2, "BlackList should be 2-D tensor");
      PADDLE_ENFORCE_EQ(bl_dims[0], black_list_len,
                        "bl_dims[0] should be equal to black_list_len");
      PADDLE_ENFORCE_EQ(bl_dims[1], 1, "bl_dims[1] should be equal to 1");
    }

    if (ctx->IsRuntime()) {
      // something to do in runtime.
    } else {
      // compile time
      ctx->SetOutputDim("Out", framework::make_ddim({-1, num_emb}));
      ctx->SetOutputDim("X_Temp_Out", x_dims);
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "W"), ctx.GetPlace());
  }
};

template <typename DeviceContext, typename T>
class CPUPyramidHashOPKernel : public framework::OpKernel<T> {
 public:
  bool should_use_term(math::bloomfilter* _filter,
                       math::bloomfilter* _black_filter, const T* word_repr,
                       int len) const {
    return (!_filter ||
            1 == math::bloomfilter_get(_filter, word_repr, len * sizeof(T))) &&
           (!_black_filter ||
            0 == math::bloomfilter_get(_black_filter, word_repr,
                                       len * sizeof(T)));
  }

  void hash_embedding_ff(const T* hash_id, int len, T* top_pos,
                         const T* weights, int _num_emb, int _rand_len,
                         int _space_len) const {
    unsigned int pos1 = XXH32(hash_id, len * sizeof(T), 0) % _space_len;
    unsigned int pos2 = XXH32(hash_id, len * sizeof(T), _rand_len) % _space_len;

    for (int j = 0; j != _num_emb; j += _rand_len) {
      if (j + _rand_len < _num_emb) {
        __builtin_prefetch(weights + pos2);
        __builtin_prefetch(top_pos + j + _rand_len);
      }

      unsigned int pos3 =
          XXH32(hash_id, len * sizeof(T), j + 2 * _rand_len) % _space_len;
      memcpy(top_pos + j, const_cast<float*>(weights + pos1),
             _rand_len * sizeof(T));
      pos1 = pos2;
      pos2 = pos3;
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom = ctx.Input<LoDTensor>("X");
    auto* _blobs_0 = ctx.Input<Tensor>("W");
    auto* _blobs_1 = ctx.Input<Tensor>("WhiteList");
    auto* _blobs_2 = ctx.Input<Tensor>("BlackList");
    auto* top = ctx.Output<LoDTensor>("Out");
    auto* drop_pos = ctx.Output<LoDTensor>("DropPos");

    int _num_emb = ctx.Attr<int>("num_emb");
    bool use_filter = ctx.Attr<bool>("use_filter");
    int white_list_len = ctx.Attr<int>("white_list_len");
    int black_list_len = ctx.Attr<int>("black_list_len");
    int _pyramid_layer = ctx.Attr<int>("pyramid_layer");
    int _is_training = ctx.Attr<int>("is_training");
    int seed = ctx.Attr<int>("seed");
    unsigned int _seed = (unsigned int)seed;
    int _rand_len = ctx.Attr<int>("rand_len");
    int _space_len = ctx.Attr<int>("space_len");
    float _drop_out_percent = ctx.Attr<float>("drop_out_percent");

    const auto& offset = bottom->lod()[0];
    const auto* bottom_data_ori = bottom->data<int32_t>();
    auto* buff = ctx.Output<LoDTensor>("X_Temp_Out");
    buff->Resize(framework::make_ddim({bottom->dims()[0], bottom->dims()[1]}));
    T* bottom_data = buff->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < bottom->dims()[0]; i++) {
      bottom_data[i] = bottom_data_ori[i];
    }

    const auto* weights = _blobs_0->data<T>();

    std::vector<size_t> top_offset;
    top_offset.resize(offset.size());
    top_offset[0] = 0;

    math::bloomfilter* _filter = NULL;
    math::bloomfilter* _black_filter = NULL;
    if (use_filter) {
      if (white_list_len != 0) {
        _filter = (math::bloomfilter*)_blobs_1->data<T>();
        PADDLE_ENFORCE_EQ(math::bloomfilter_check(_filter), 1,
                          "white filter not load");
      }
      if (black_list_len != 0) {
        _black_filter = (math::bloomfilter*)_blobs_2->data<T>();
        PADDLE_ENFORCE_EQ(math::bloomfilter_check(_black_filter), 1,
                          "black filter not load");
      }
    }

    drop_pos->Resize(framework::make_ddim(
        {bottom->dims()[0] * bottom->dims()[1] * _pyramid_layer, 1}));
    std::vector<size_t> drop_pos_offset;
    drop_pos_offset.resize(offset.size());
    drop_pos_offset[0] = 0;
    int* iter = drop_pos->mutable_data<int>(ctx.GetPlace());
    int* iter_end = iter;

    for (size_t i = 0; i < top_offset.size() - 1; ++i) {
      int w = offset[i + 1] - offset[i];
      int nsentense_with_pyramid = 0;
      if (w < 2) {
        nsentense_with_pyramid = 0;
      } else {
        for (int ilayer = 1; ilayer < _pyramid_layer && ilayer < w; ++ilayer) {
          for (int l = 0; l < w - ilayer; ++l) {
            if (should_use_term(_filter, _black_filter,
                                (const T*)(bottom_data + offset[i] + l),
                                ilayer + 1)) {
              if (_is_training != 0) {
                unsigned int rand_val = rand_r(&_seed);
                T rate = static_cast<T>(rand_val) / (RAND_MAX);
                *(iter_end++) = (rate < _drop_out_percent ? 0 : 1);
              } else {
                *(iter_end++) = 1;
              }
            } else {
              *(iter_end++) = 0;
            }
          }
        }
        nsentense_with_pyramid = std::count(iter, iter_end, 1);
        iter = iter_end;
      }
      drop_pos_offset[i + 1] = drop_pos_offset[i] + nsentense_with_pyramid;
      top_offset[i + 1] =
          top_offset[i] +
          (nsentense_with_pyramid == 0 ? 1 : nsentense_with_pyramid);
    }

    int top_l = top_offset[top_offset.size() - 1];

    framework::LoD top_lod;
    top_lod.push_back(top_offset);
    top->set_lod(top_lod);
    top->Resize(framework::make_ddim({top_l, _num_emb}));
    auto* top_data = top->mutable_data<T>(ctx.GetPlace());

    framework::LoD drop_pos_lod;
    drop_pos_lod.push_back(drop_pos_offset);
    drop_pos->set_lod(drop_pos_lod);

    iter = drop_pos->mutable_data<int>(ctx.GetPlace());
    int top_counter = 0;
    for (size_t i = 0; i < offset.size() - 1; ++i) {
      int w_drop = drop_pos_offset[i + 1] - drop_pos_offset[i];
      int w = offset[i + 1] - offset[i];
      if (w_drop == 0) {
        if (w >= 2) {
          for (int ilayer = 1; ilayer < _pyramid_layer && ilayer < w;
               ++ilayer) {
            for (int l = 0; l < w - ilayer; ++l) {
              iter++;
            }
          }
        }
        auto* top_pos = top_data + top_counter++ * _num_emb;
        memset(top_pos, 0, _num_emb * sizeof(T));
        continue;
      }
      if (w >= 2) {
        for (int ilayer = 1; ilayer < _pyramid_layer && ilayer < w; ++ilayer) {
          for (int l = 0; l < w - ilayer; ++l) {
            if (*(iter++) == 0) {
              // do nothing
            } else {
              auto* top_pos = top_data + top_counter++ * _num_emb;
              hash_embedding_ff((const T*)(bottom_data + offset[i] + l),
                                ilayer + 1, top_pos, weights, _num_emb,
                                _rand_len, _space_len);
            }
          }
        }
      }
    }
    if (iter != iter_end) {
      exit(1);
    }
    if (_is_training == 0) {
      avx_axpy_noadd(top_data, top_data, top->dims()[0] * top->dims()[1],
                     _drop_out_percent);
    }
  }
};

class PyramidHashOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true, "Input(X) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true, "Input(W) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("DropPos"), true,
                      "Input(DropPos) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("X_Temp_Out"), true,
                      "Input(X_Temp_Out) should not be null.");
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        "Input(Out@GRAD) of PyramidHashGradOp should not be null.");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "W"), ctx.GetPlace());
  }
};

template <typename T>
class PyramidHashGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto* op_desc_ptr = new T();
    op_desc_ptr->SetType("pyramid_hash_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    op_desc_ptr->SetInput("W", this->Input("W"));
    op_desc_ptr->SetInput("DropPos", this->Output("DropPos"));
    op_desc_ptr->SetInput("X_Temp_Out", this->Output("X_Temp_Out"));

    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op_desc_ptr->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(op_desc_ptr);
  }
};

template <typename DeviceContext, typename T>
class CPUPyramidHashOPGradKernel : public framework::OpKernel<T> {
 public:
  void hash_embedding_bp(const T* hash_id, int len, const T* top_pos,
                         T* weights, T mlr, int _num_emb, int _rand_len,
                         int _space_len) const {
    for (int j = 0; j != _num_emb; j += _rand_len) {
      unsigned int pos = XXH32(hash_id, len * sizeof(T), j) % _space_len;
      avx_axpy(top_pos + j, weights + pos, _rand_len, mlr);
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom = ctx.Input<LoDTensor>("X");
    auto* _blobs = ctx.Input<Tensor>("W");
    auto* drop_pos = ctx.Input<LoDTensor>("DropPos");
    auto* top = ctx.Input<LoDTensor>(framework::GradVarName("Out"));

    int _num_emb = ctx.Attr<int>("num_emb");
    float _lr = ctx.Attr<float>("lr");
    int _rand_len = ctx.Attr<int>("rand_len");
    int _space_len = ctx.Attr<int>("space_len");
    int _pyramid_layer = ctx.Attr<int>("pyramid_layer");

    auto* buff = ctx.Input<LoDTensor>("X_Temp_Out");
    auto* bottom_data = buff->data<T>();

    int _slot_len = bottom->dims()[0];
    if (static_cast<size_t>(_slot_len) == bottom->lod()[0].size() - 1 &&
        std::count(bottom_data, bottom_data + _slot_len, -1) == _slot_len) {
      return;
    }

    auto& offset = bottom->lod()[0];
    auto& drop_pos_offset = drop_pos->lod()[0];

    const auto* top_diff = top->data<T>();
    T* weights = const_cast<T*>(_blobs->data<T>());
    T mlr = -1.0 * _lr;

    const int* iter = drop_pos->data<int>();
    int top_counter = 0;
    for (size_t i = 0; i < offset.size() - 1; ++i) {
      int w = offset[i + 1] - offset[i];
      int w_drop = drop_pos_offset[i + 1] - drop_pos_offset[i];
      if (w_drop == 0) {
        top_counter++;
      }
      if (w > 1) {
        for (int ilayer = 1; ilayer < _pyramid_layer && ilayer < w; ++ilayer) {
          for (int l = 0; l < w - ilayer; ++l) {
            if (*(iter++) == 0) {
              // do nothing
            } else {
              const T* top_pos = top_diff + top_counter++ * _num_emb;
              hash_embedding_bp((const T*)(bottom_data + offset[i] + l),
                                ilayer + 1, top_pos, weights, mlr, _num_emb,
                                _rand_len, _space_len);
            }
          }
        }
      } else {
        // do nothing
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(pyramid_hash, ops::PyramidHashOP, ops::PyramidHashOpMaker,
                  ops::PyramidHashGradOpMaker<paddle::framework::OpDesc>,
                  ops::PyramidHashGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(pyramid_hash_grad, ops::PyramidHashOpGrad);

REGISTER_OP_CPU_KERNEL(
    pyramid_hash, ops::CPUPyramidHashOPKernel<plt::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    pyramid_hash_grad,
    ops::CPUPyramidHashOPGradKernel<plt::CPUDeviceContext, float>);
