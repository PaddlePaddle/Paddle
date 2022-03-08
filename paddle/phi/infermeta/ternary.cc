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

#include "paddle/phi/infermeta/ternary.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

void AddmmInferMeta(const MetaTensor& input,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    float alpha,
                    float beta,
                    MetaTensor* out) {
  auto input_dims = input.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  auto ndim_input = input_dims.size();
  auto ndim_x = x_dims.size();
  auto ndim_y = y_dims.size();

  VLOG(3) << "addmm operator input.shape=" << input_dims
          << " x.shape=" << x_dims << " y.shape=" << y_dims << " beta=" << beta
          << " alpha=" << alpha << " ndim_input=" << ndim_input
          << " ndim_x=" << ndim_x << " ndim_y=" << ndim_y;

  PADDLE_ENFORCE_NE(
      product(input_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'input' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));

  PADDLE_ENFORCE_NE(
      product(x_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'x' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));

  PADDLE_ENFORCE_NE(
      product(y_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'y' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));
  // dim check
  PADDLE_ENFORCE_EQ(
      ndim_input,
      2,
      errors::InvalidArgument("The input tensor input's dimension must be 2. "
                              "But received input's dimension = [%s].",
                              ndim_input));
  PADDLE_ENFORCE_EQ(
      ndim_x,
      2,
      errors::InvalidArgument("The input tensor x's dimension must be 2. "
                              "But received x's dimension = [%s].",
                              ndim_x));
  PADDLE_ENFORCE_EQ(
      ndim_y,
      2,
      errors::InvalidArgument("The input tensor y's dimension must be 2. "
                              "But received y's dimension = [%s].",
                              ndim_y));

  std::vector<int64_t> output_dims;
  output_dims.push_back(x_dims[0]);
  output_dims.push_back(y_dims[1]);

  out->set_dims(make_ddim(output_dims));
  out->share_lod(input);
  out->set_dtype(input.dtype());
}

void ScatterInferMeta(const MetaTensor& x,
                      const MetaTensor& index,
                      const MetaTensor& updates,
                      bool overwrite,
                      MetaTensor* out) {
  const auto& updates_dims = updates.dims();
  const auto& ref_dims = x.dims();
  const auto& index_dims = index.dims();
  PADDLE_ENFORCE_EQ(
      index_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "The size of Input(Ids)'s shape should be equal to 1, but "
          "received the rank of Input(Ids) is %d.",
          index_dims.size()));
  PADDLE_ENFORCE_EQ(
      ref_dims.size(),
      updates_dims.size(),
      phi::errors::InvalidArgument(
          "Input(X) and Input(Updates) should have the same shape size, "
          "but received the size of Input(x)'s shape is %d, the size of "
          "Input(Updates)'s shape is %d.",
          ref_dims.size(),
          updates_dims.size()));
  PADDLE_ENFORCE_EQ(
      updates_dims[0],
      index_dims[0],
      phi::errors::InvalidArgument(
          "Input(Updates) and Input(Ids) should have same batch-size, but"
          " received Input(Updates)'s batch-size is %d, Input(Ids)'s "
          "batch-size is %d.",
          updates_dims[0],
          index_dims[0]));
  out->set_dims(ref_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void ScatterNdAddInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& updates,
                           MetaTensor* out) {
  const auto& ref_dims = x.dims();
  auto ref_dims_size = ref_dims.size();
  const auto& index_dims = index.dims();
  auto index_dims_size = index_dims.size();
  const auto& updates_dims = updates.dims();
  auto updates_dims_size = updates_dims.size();

  PADDLE_ENFORCE_LE(
      index_dims[index_dims_size - 1],
      ref_dims_size,
      phi::errors::InvalidArgument(
          "The last dimension of Input(Index)'s shape should be no greater "
          "than the rank of Input(X), but received the last dimension of "
          "Input(Index)'s shape is %d, the rank of Input(X) is %d.",
          index_dims[index_dims_size - 1],
          ref_dims_size));
  PADDLE_ENFORCE_GE(index_dims_size,
                    2UL,
                    phi::errors::InvalidArgument(
                        "The rank of Input(Index) should be greater than 1, "
                        "but received the rank of Input(Index) is %d.",
                        index_dims_size));

  // update.shape = index.shape[:-1] + output.shape[index.shape[-1]:]
  std::vector<int64_t> r_updates_dims;
  for (int64_t i = 0; i < index_dims_size - 1; ++i) {
    r_updates_dims.emplace_back(index_dims[i]);
  }
  for (int64_t i = index_dims[index_dims_size - 1]; i < ref_dims_size; ++i) {
    r_updates_dims.emplace_back(ref_dims[i]);
  }

  PADDLE_ENFORCE_EQ(
      r_updates_dims.size(),
      updates_dims_size,
      phi::errors::InvalidArgument(
          "Updates has wrong shape. The shape of Updates and Input(Updates) "
          "should be same, but received the shape of Updates is %d, "
          "the shape of Input(Updates) is %d.",
          r_updates_dims.size(),
          updates_dims_size));

  for (int64_t i = 0; i < updates_dims_size; ++i) {
    PADDLE_ENFORCE_EQ(
        r_updates_dims[i],
        updates_dims[i],
        phi::errors::InvalidArgument(
            "Updates has wrong shape. The dimensions of Updates and "
            "Input(Updates) should match, but received Updates's"
            "%d-th dimension is %d, Input(Updates)'s %d-th "
            "dimension is %d.",
            i,
            r_updates_dims[i],
            i,
            updates_dims[i]));
  }
  out->set_dims(ref_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void ViterbiDecodeInferMeta(const MetaTensor& input,
                            const MetaTensor& transition,
                            const MetaTensor& length,
                            bool include_bos_eos_tag,
                            MetaTensor* scores,
                            MetaTensor* path,
                            MetaConfig config) {
  auto in_dims = input.dims();
  PADDLE_ENFORCE_EQ(in_dims.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The rank of Input in ViterbiDecode  must be 3. But "
                        "received Input's rank is %d.",
                        in_dims.size()));
  auto length_dims = length.dims();
  PADDLE_ENFORCE_EQ(length_dims.size(),
                    1,
                    phi::errors::InvalidArgument(
                        "The rank of Length in ViterbiDecode must be 1. But "
                        "received Length's rank is %d.",
                        length_dims.size()));
  auto transition_dims = transition.dims();
  PADDLE_ENFORCE_EQ(
      transition_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The rank of Transition in ViterbiDecode must be 2. But "
          "received Transition's rank is %d.",
          transition_dims.size()));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        in_dims[0],
        length_dims[0],
        phi::errors::InvalidArgument(
            "The batch size of Input and Length should be equal."));
    PADDLE_ENFORCE_EQ(in_dims[2],
                      transition_dims[0],
                      phi::errors::InvalidArgument(
                          "The number of tags of Input (%d) and Transition "
                          "(%d) should be equal.",
                          transition_dims[0],
                          in_dims[2]));
  }
  scores->set_dims(length_dims);
  scores->set_dtype(length.dtype());
}

void LerpInferMeta(const MetaTensor& x,
                   const MetaTensor& y,
                   const MetaTensor& weight,
                   MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto w_dims = weight.dims();
  DDim out_dims;
  out_dims = funcs::GetOutputDims(x_dims, y_dims);
  if (w_dims.size() > 1 || w_dims[0] != 1) {
    out_dims = funcs::GetOutputDims(out_dims, w_dims);
  }
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void LinspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       MetaTensor* out) {
  auto s_dims = start.dims();
  PADDLE_ENFORCE_EQ(
      (s_dims.size() == 1) && (s_dims[0] == 1),
      true,
      phi::errors::InvalidArgument("The shape of Input(Start) must be [1],"
                                   "but received input shape is [%s].",
                                   s_dims));
  auto e_dims = stop.dims();
  PADDLE_ENFORCE_EQ(
      (e_dims.size() == 1) && (e_dims[0] == 1),
      true,
      phi::errors::InvalidArgument("The shape of Input(Stop) must be [1],"
                                   "but received input shape is [%s].",
                                   e_dims));
  auto step_dims = number.dims();
  PADDLE_ENFORCE_EQ(
      (step_dims.size() == 1) && (step_dims[0] == 1),
      true,
      phi::errors::InvalidArgument("The shape of Input(Num) must be [1],"
                                   "but received input shape is [%s].",
                                   step_dims));
  out->set_dims(phi::make_ddim({-1}));
  out->set_dtype(start.dtype());
}

void GraphSendRecvInferMeta(const MetaTensor& x,
                            const MetaTensor& src_index,
                            const MetaTensor& dst_index,
                            const std::string& pool_type,
                            MetaTensor* out,
                            MetaTensor* dst_count) {
  auto src_index_dims = src_index.dims();
  if (src_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(src_index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of Src_index should be 1 when it "
                          "is 2D, but we get %d",
                          src_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        src_index_dims.size(),
        1,
        phi::errors::InvalidArgument(
            "The Src_index should be 1D, when it is not 2D, but we get %d",
            src_index_dims.size()));
  }

  auto dst_index_dims = dst_index.dims();
  if (dst_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(dst_index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of Dst_index should be 1 when it "
                          "is 2D, but we get %d",
                          dst_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        dst_index_dims.size(),
        1,
        phi::errors::InvalidArgument("The Dst_index should be 1D, "
                                     "when it is not 2D, but we get %d",
                                     dst_index_dims.size()));
  }

  PADDLE_ENFORCE_EQ(src_index_dims[0],
                    dst_index_dims[0],
                    phi::errors::InvalidArgument(
                        "Src_index and Dst_index should have the same shape."));

  auto dims = x.dims();
  out->set_dims(dims);
  out->set_dtype(x.dtype());

  if (pool_type == "MEAN") {
    dst_count->set_dims({dims[0]});
    dst_count->set_dtype(DataType::INT32);
  }
}

void RangeInferMeta(const MetaTensor& Start,
                    const MetaTensor& End,
                    const MetaTensor& Step,
                    MetaTensor* out) {
  auto start_dims = Start.dims();
  auto end_dims = End.dims();
  auto step_dims = Step.dims();
  PADDLE_ENFORCE_EQ(
      start_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "The dim of the shape of Input(Start) should be 1, but got %d",
          start_dims.size()));

  PADDLE_ENFORCE_EQ(start_dims[0],
                    1,
                    phi::errors::InvalidArgument(
                        "The first dim of the shape of Input(Start) should "
                        "be 1, but got %d",
                        start_dims[0]));
  PADDLE_ENFORCE_EQ(
      end_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "The dim of the shape of Input(End) should be 1, but got %d",
          end_dims.size()));

  PADDLE_ENFORCE_EQ(
      end_dims[0],
      1,
      phi::errors::InvalidArgument("The first dim of the shape of "
                                   "Input(End) should be 1, but got %d",
                                   end_dims[0]));
  PADDLE_ENFORCE_EQ(
      step_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "The dim of the shape of Input(Step) should be 1, but got %d",
          step_dims.size()));

  PADDLE_ENFORCE_EQ(step_dims[0],
                    1,
                    phi::errors::InvalidArgument(
                        "The first dim of the shape of Input(Step) should "
                        "be 1, but got %d",
                        step_dims[0]));
  out->set_dims({-1});
}

}  // namespace phi
