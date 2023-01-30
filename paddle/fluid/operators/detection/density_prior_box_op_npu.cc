/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection/density_prior_box_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

<<<<<<< HEAD
=======
using Tensor = framework::Tensor;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
using fp16 = paddle::platform::float16;

template <typename T>
struct DensityPriorBoxFunction {
 public:
  explicit DensityPriorBoxFunction(const framework::ExecutionContext& ctx)
      : ctx(ctx) {
    place = ctx.GetPlace();
    stream = ctx.template device_context<platform::NPUDeviceContext>().stream();
    t0.mutable_data<float>({1}, place);
    t1.mutable_data<float>({1}, place);
    tn.mutable_data<float>({1}, place);
    FillNpuTensorWithConstant<float>(&t0, static_cast<float>(0));
    FillNpuTensorWithConstant<float>(&t1, static_cast<float>(1));
  }
<<<<<<< HEAD
  void Arange(int n, phi::DenseTensor* x) {
=======
  void Arange(int n, Tensor* x) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  x should be init first
    FillNpuTensorWithConstant<float>(&tn, static_cast<float>(n));
    const auto& runner = NpuOpRunner("Range", {t0, tn, t1}, {*x}, {});
    runner.Run(stream);
  }
<<<<<<< HEAD
  void Add(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
=======
  void Add(const Tensor* x, const Tensor* y, Tensor* z) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  z should be init first
    const auto& runner = NpuOpRunner("AddV2", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
<<<<<<< HEAD
  void Cast(const phi::DenseTensor* x, phi::DenseTensor* y) {
=======
  void Cast(const Tensor* x, Tensor* y) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    auto dst_dtype =
        ConvertToNpuDtype(framework::TransToProtoVarType(y->type()));
    const auto& runner = NpuOpRunner(
        "Cast", {*x}, {*y}, {{"dst_type", static_cast<int>(dst_dtype)}});
    runner.Run(stream);
  }
<<<<<<< HEAD
  void Sub(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
=======
  void Sub(const Tensor* x, const Tensor* y, Tensor* z) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  z should be init first
    const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
<<<<<<< HEAD
  void Mul(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
=======
  void Mul(const Tensor* x, const Tensor* y, Tensor* z) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  y should be init first
    const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
<<<<<<< HEAD
  void Adds(const phi::DenseTensor* x, float scalar, phi::DenseTensor* y) {
=======
  void Adds(const Tensor* x, float scalar, Tensor* y) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  y should be init first
    const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scalar}});
    runner.Run(stream);
  }
<<<<<<< HEAD
  void Muls(const phi::DenseTensor* x, float scalar, phi::DenseTensor* y) {
=======
  void Muls(const Tensor* x, float scalar, Tensor* y) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  y should be init first
    const auto& runner = NpuOpRunner("Muls", {*x}, {*y}, {{"value", scalar}});
    runner.Run(stream);
  }
<<<<<<< HEAD
  void Maximum(const phi::DenseTensor* x,
               const phi::DenseTensor* y,
               phi::DenseTensor* z) {
=======
  void Maximum(const Tensor* x, const Tensor* y, Tensor* z) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  y should be init first
    const auto& runner = NpuOpRunner("Maximum", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
<<<<<<< HEAD
  void Minimum(const phi::DenseTensor* x,
               const phi::DenseTensor* y,
               phi::DenseTensor* z) {
=======
  void Minimum(const Tensor* x, const Tensor* y, Tensor* z) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  y should be init first
    const auto& runner = NpuOpRunner("Minimum", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
<<<<<<< HEAD
  void Concat(const std::vector<phi::DenseTensor>& inputs,
              int axis,
              phi::DenseTensor* output) {
=======
  void Concat(const std::vector<Tensor>& inputs, int axis, Tensor* output) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  output should be init first
    std::vector<std::string> names;
    for (size_t i = 0; i < inputs.size(); i++) {
      names.push_back("x" + std::to_string(i));
    }
    NpuOpRunner runner{
        "ConcatD",
        {inputs},
        {*output},
        {{"concat_dim", axis}, {"N", static_cast<int>(inputs.size())}}};
    runner.AddInputNames(names);
    runner.Run(stream);
  }
<<<<<<< HEAD
  void Tile(const phi::DenseTensor* x,
            phi::DenseTensor* y,
            const std::vector<int>& multiples) {
=======
  void Tile(const Tensor* x, Tensor* y, const std::vector<int>& multiples) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  y should be init first
    if (x->dims() == y->dims()) {
      framework::TensorCopy(
          *x,
          place,
          ctx.template device_context<platform::NPUDeviceContext>(),
          y);
      return;
    }
    const auto& runner =
        NpuOpRunner("TileD", {*x}, {*y}, {{"multiples", multiples}});
    runner.Run(stream);
  }
<<<<<<< HEAD
  void FloatVec2Tsr(const std::vector<float>& vec, phi::DenseTensor* tsr_dst) {
=======
  void FloatVec2Tsr(const std::vector<float>& vec, Tensor* tsr_dst) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //
    framework::TensorFromVector<T>(vec, ctx.device_context(), tsr_dst);
    ctx.template device_context<platform::NPUDeviceContext>().Wait();
  }

 private:
  platform::Place place;
  aclrtStream stream;
  const framework::ExecutionContext& ctx;
<<<<<<< HEAD
  phi::DenseTensor t0;
  phi::DenseTensor t1;
  phi::DenseTensor tn;
};

template <>
void DensityPriorBoxFunction<fp16>::Arange(int n, phi::DenseTensor* x) {
  phi::DenseTensor x_fp32(experimental::DataType::FLOAT32);
=======
  Tensor t0;
  Tensor t1;
  Tensor tn;
};

template <>
void DensityPriorBoxFunction<fp16>::Arange(int n, Tensor* x) {
  Tensor x_fp32(experimental::DataType::FLOAT32);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  x_fp32.mutable_data<float>(x->dims(), place);
  FillNpuTensorWithConstant<float>(&tn, static_cast<float>(n));
  const auto& runner = NpuOpRunner("Range", {t0, tn, t1}, {x_fp32}, {});
  runner.Run(stream);
  Cast(&x_fp32, x);
}

template <>
void DensityPriorBoxFunction<fp16>::FloatVec2Tsr(const std::vector<float>& vec,
<<<<<<< HEAD
                                                 phi::DenseTensor* tsr_dst) {
  phi::DenseTensor tsr_fp32(experimental::DataType::FLOAT32);
=======
                                                 Tensor* tsr_dst) {
  Tensor tsr_fp32(experimental::DataType::FLOAT32);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  tsr_fp32.mutable_data<float>(tsr_dst->dims(), place);
  framework::TensorFromVector<float>(vec, ctx.device_context(), &tsr_fp32);
  ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
  Cast(&tsr_fp32, tsr_dst);
}

template <typename T>
class DensityPriorBoxOpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
<<<<<<< HEAD
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* image = ctx.Input<phi::DenseTensor>("Image");
    auto* boxes = ctx.Output<phi::DenseTensor>("Boxes");
    auto* vars = ctx.Output<phi::DenseTensor>("Variances");
=======
    auto* input = ctx.Input<paddle::framework::Tensor>("Input");
    auto* image = ctx.Input<paddle::framework::Tensor>("Image");
    auto* boxes = ctx.Output<paddle::framework::Tensor>("Boxes");
    auto* vars = ctx.Output<paddle::framework::Tensor>("Variances");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    auto variances = ctx.Attr<std::vector<float>>("variances");
    auto clip = ctx.Attr<bool>("clip");

    auto fixed_sizes = ctx.Attr<std::vector<float>>("fixed_sizes");
    auto fixed_ratios = ctx.Attr<std::vector<float>>("fixed_ratios");
    auto densities = ctx.Attr<std::vector<int>>("densities");

    float step_w = ctx.Attr<float>("step_w");
    float step_h = ctx.Attr<float>("step_h");
    float offset = ctx.Attr<float>("offset");

    int image_w = image->dims()[3];
    int image_h = image->dims()[2];
    int layer_w = input->dims()[3];
    int layer_h = input->dims()[2];

    auto _type = input->dtype();
    auto place = ctx.GetPlace();
    DensityPriorBoxFunction<T> F(ctx);

<<<<<<< HEAD
    phi::DenseTensor h(_type);
    h.mutable_data<T>({layer_h}, place);
    phi::DenseTensor w(_type);
=======
    Tensor h(_type);
    h.mutable_data<T>({layer_h}, place);
    Tensor w(_type);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    w.mutable_data<T>({layer_w}, place);
    F.Arange(layer_h, &h);
    F.Arange(layer_w, &w);
    h.Resize({layer_h, 1, 1, 1});
    w.Resize({1, layer_w, 1, 1});

    step_w = step_w > 0 ? step_w : static_cast<float>(image_w) / layer_w;
    step_h = step_h > 0 ? step_h : static_cast<float>(image_h) / layer_h;
    int step_average = static_cast<int>((step_w + step_h) * 0.5);

    int ratios_size = fixed_ratios.size();
    int num_priors_per_ratio = 0;
    for (size_t i = 0; i < densities.size(); ++i) {
      num_priors_per_ratio += densities[i] * densities[i];
    }
<<<<<<< HEAD
    phi::DenseTensor di(_type);
    phi::DenseTensor dj(_type);
    phi::DenseTensor shifts(_type);
    phi::DenseTensor box_w_ratio(_type);
    phi::DenseTensor box_h_ratio(_type);
=======
    Tensor di(_type);
    Tensor dj(_type);
    Tensor shifts(_type);
    Tensor box_w_ratio(_type);
    Tensor box_h_ratio(_type);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    di.mutable_data<T>({ratios_size * num_priors_per_ratio}, place);
    dj.mutable_data<T>({ratios_size * num_priors_per_ratio}, place);
    shifts.mutable_data<T>({ratios_size * num_priors_per_ratio}, place);
    box_w_ratio.mutable_data<T>({ratios_size * num_priors_per_ratio}, place);
    box_h_ratio.mutable_data<T>({ratios_size * num_priors_per_ratio}, place);

    int64_t start = 0;
    std::vector<int> vec_tile = {0, 0, 0};
    for (size_t i = 0; i < densities.size(); ++i) {
      //  Range = start:start+ratios_size*density_sqr, density = densities[i]
      int density_sqr = densities[i] * densities[i];
      //  shifts[Range] = [step_average/density]*ratios_size*density_sqr
<<<<<<< HEAD
      phi::DenseTensor shifts_part =
=======
      Tensor shifts_part =
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
          shifts.Slice(start, start + ratios_size * density_sqr);
      FillNpuTensorWithConstant<T>(&shifts_part,
                                   static_cast<T>(step_average / densities[i]));

      //  di[Range] = [ i // density for i in range(density_sqr) ] * ratios_size
      //  dj[Range] = [ i % density for i in range(density_sqr) ] * ratios_size
<<<<<<< HEAD
      phi::DenseTensor di_part =
          di.Slice(start, start + ratios_size * density_sqr);
      phi::DenseTensor dj_part =
          dj.Slice(start, start + ratios_size * density_sqr);
      if (densities[i] > 1) {
        di_part.Resize({ratios_size, densities[i], densities[i]});
        dj_part.Resize({ratios_size, densities[i], densities[i]});
        phi::DenseTensor range_n(_type);
=======
      Tensor di_part = di.Slice(start, start + ratios_size * density_sqr);
      Tensor dj_part = dj.Slice(start, start + ratios_size * density_sqr);
      if (densities[i] > 1) {
        di_part.Resize({ratios_size, densities[i], densities[i]});
        dj_part.Resize({ratios_size, densities[i], densities[i]});
        Tensor range_n(_type);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        range_n.mutable_data<T>({densities[i]}, place);
        F.Arange(densities[i], &range_n);
        range_n.Resize({1, densities[i], 1});
        vec_tile[0] = ratios_size;
        vec_tile[1] = 1;
        vec_tile[2] = densities[i];
        F.Tile(&range_n, &di_part, vec_tile);
        range_n.Resize({1, 1, densities[i]});
        vec_tile[1] = densities[i];
        vec_tile[2] = 1;
        F.Tile(&range_n, &dj_part, vec_tile);
      } else {
        FillNpuTensorWithConstant<T>(&di_part, static_cast<T>(0));
        FillNpuTensorWithConstant<T>(&dj_part, static_cast<T>(0));
      }

      int start_box_ratio = start;
      for (float ar : fixed_ratios) {
        //  Range_mini = start_box_ratio:start_box_ratio+density_sqr
        //  box_h_ratio[Range_mini] = [fixed_sizes[i] * sqrt(ar)]  * density_sqr
        //  box_w_ratio[Range_mini] = [fixed_sizes[i] / sqrt(ar)]  * density_sqr
<<<<<<< HEAD
        phi::DenseTensor box_h_ratio_part =
            box_h_ratio.Slice(start_box_ratio, start_box_ratio + density_sqr);
        phi::DenseTensor box_w_ratio_part =
=======
        Tensor box_h_ratio_part =
            box_h_ratio.Slice(start_box_ratio, start_box_ratio + density_sqr);
        Tensor box_w_ratio_part =
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            box_w_ratio.Slice(start_box_ratio, start_box_ratio + density_sqr);
        FillNpuTensorWithConstant<T>(&box_w_ratio_part,
                                     static_cast<T>(fixed_sizes[i] * sqrt(ar)));
        FillNpuTensorWithConstant<T>(&box_h_ratio_part,
                                     static_cast<T>(fixed_sizes[i] / sqrt(ar)));
        start_box_ratio += density_sqr;
      }
      start = start_box_ratio;
    }
    di.Resize({1, 1, ratios_size * num_priors_per_ratio, 1});
    dj.Resize({1, 1, ratios_size * num_priors_per_ratio, 1});
    shifts.Resize({1, 1, ratios_size * num_priors_per_ratio, 1});
    box_w_ratio.Resize({1, 1, ratios_size * num_priors_per_ratio, 1});
    box_h_ratio.Resize({1, 1, ratios_size * num_priors_per_ratio, 1});

    //  c_x = (w+offset)*step_w - 0.5*step_average + 0.5*shifts + dj*shifts
    //  c_y = (h+offset)*step_h - 0.5*step_average + 0.5*shifts + di*shifts
<<<<<<< HEAD
    phi::DenseTensor c_x(_type);
    phi::DenseTensor c_y(_type);
=======
    Tensor c_x(_type);
    Tensor c_y(_type);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    auto dim0 =
        phi::make_ddim({1, layer_w, ratios_size * num_priors_per_ratio, 1});
    auto dim1 =
        phi::make_ddim({layer_h, 1, ratios_size * num_priors_per_ratio, 1});
    c_x.mutable_data<T>(dim0, place);
    c_y.mutable_data<T>(dim1, place);
    F.Adds(&w, offset, &w);
    F.Muls(&w, step_w, &w);
    F.Adds(&w, static_cast<float>(-step_average) * static_cast<float>(0.5), &w);
    F.Adds(&h, offset, &h);
    F.Muls(&h, step_h, &h);
    F.Adds(&h, static_cast<float>(-step_average) * static_cast<float>(0.5), &h);
    F.Mul(&di, &shifts, &di);
    F.Mul(&dj, &shifts, &dj);
    F.Muls(&shifts, static_cast<float>(0.5), &shifts);
    F.Add(&di, &shifts, &di);
    F.Add(&dj, &shifts, &dj);
    F.Add(&dj, &w, &c_x);
    F.Add(&di, &h, &c_y);

    //  box_w_ratio = box_w_ratio / 2
    //  box_h_ratio = box_h_ratio / 2
    F.Muls(&box_w_ratio, static_cast<float>(0.5), &box_w_ratio);
    F.Muls(&box_h_ratio, static_cast<float>(0.5), &box_h_ratio);

<<<<<<< HEAD
    phi::DenseTensor zero_t(_type);
    phi::DenseTensor one_t(_type);
=======
    Tensor zero_t(_type);
    Tensor one_t(_type);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    zero_t.mutable_data<T>({1}, place);
    one_t.mutable_data<T>({1}, place);
    FillNpuTensorWithConstant<T>(&zero_t, static_cast<T>(0));
    FillNpuTensorWithConstant<T>(&one_t, static_cast<T>(1));

<<<<<<< HEAD
    phi::DenseTensor outbox0(_type);
    phi::DenseTensor outbox1(_type);
    phi::DenseTensor outbox2(_type);
    phi::DenseTensor outbox3(_type);
=======
    Tensor outbox0(_type);
    Tensor outbox1(_type);
    Tensor outbox2(_type);
    Tensor outbox3(_type);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    outbox0.mutable_data<T>(dim0, place);
    outbox1.mutable_data<T>(dim1, place);
    outbox2.mutable_data<T>(dim0, place);
    outbox3.mutable_data<T>(dim1, place);

    //  outbox0 = max ( (c_x - box_w_ratio)/image_w, 0 )
    //  outbox1 = max ( (c_y - box_h_ratio)/image_h, 0 )
    //  outbox2 = min ( (c_x + box_w_ratio)/image_w, 1 )
    //  outbox3 = min ( (c_y + box_h_ratio)/image_h, 1 )
    F.Sub(&c_x, &box_w_ratio, &outbox0);
    F.Sub(&c_y, &box_h_ratio, &outbox1);
    F.Add(&c_x, &box_w_ratio, &outbox2);
    F.Add(&c_y, &box_h_ratio, &outbox3);
    F.Muls(&outbox0, static_cast<float>(1.0 / image_w), &outbox0);
    F.Muls(&outbox1, static_cast<float>(1.0 / image_h), &outbox1);
    F.Muls(&outbox2, static_cast<float>(1.0 / image_w), &outbox2);
    F.Muls(&outbox3, static_cast<float>(1.0 / image_h), &outbox3);

    F.Maximum(&outbox0, &zero_t, &outbox0);
    F.Maximum(&outbox1, &zero_t, &outbox1);
    F.Minimum(&outbox2, &one_t, &outbox2);
    F.Minimum(&outbox3, &one_t, &outbox3);
    if (clip) {
      //  outbox0 = min ( outbox0, 1 )
      //  outbox1 = min ( outbox1, 1 )
      //  outbox2 = max ( outbox2, 0 )
      //  outbox3 = max ( outbox3, 0 )
      F.Minimum(&outbox0, &one_t, &outbox0);
      F.Minimum(&outbox1, &one_t, &outbox1);
      F.Maximum(&outbox2, &zero_t, &outbox2);
      F.Maximum(&outbox3, &zero_t, &outbox3);
    }

    auto out_dim = phi::make_ddim(
        {layer_h, layer_w, ratios_size * num_priors_per_ratio, 4});
    boxes->mutable_data<T>(place);
    vars->mutable_data<T>(place);
<<<<<<< HEAD
    phi::DenseTensor boxes_share(_type);
    phi::DenseTensor vars_share(_type);
=======
    Tensor boxes_share(_type);
    Tensor vars_share(_type);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    boxes_share.ShareDataWith(*boxes);
    boxes_share.Resize(out_dim);
    vars_share.ShareDataWith(*vars);
    vars_share.Resize(out_dim);

<<<<<<< HEAD
    phi::DenseTensor box0(_type);
    phi::DenseTensor box1(_type);
    phi::DenseTensor box2(_type);
    phi::DenseTensor box3(_type);
=======
    Tensor box0(_type);
    Tensor box1(_type);
    Tensor box2(_type);
    Tensor box3(_type);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    // out_dim = {layer_h, layer_w, ratios_size*num_priors_per_ratio, 1}
    out_dim[3] = 1;
    box0.mutable_data<T>(out_dim, place);
    box1.mutable_data<T>(out_dim, place);
    box2.mutable_data<T>(out_dim, place);
    box3.mutable_data<T>(out_dim, place);

    std::vector<int> vec_exp_out02 = {layer_h, 1, 1, 1};
    std::vector<int> vec_exp_out13 = {1, layer_w, 1, 1};
    F.Tile(&outbox0, &box0, vec_exp_out02);
    F.Tile(&outbox1, &box1, vec_exp_out13);
    F.Tile(&outbox2, &box2, vec_exp_out02);
    F.Tile(&outbox3, &box3, vec_exp_out13);
    F.Concat({box0, box1, box2, box3}, 3, &boxes_share);

    std::vector<int> multiples = {
        layer_h, layer_w, ratios_size * num_priors_per_ratio, 1};
<<<<<<< HEAD
    phi::DenseTensor variances_t(_type);
=======
    Tensor variances_t(_type);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    //  variances.size() == 4
    variances_t.mutable_data<T>({4}, place);
    F.FloatVec2Tsr(variances, &variances_t);
    F.Tile(&variances_t, &vars_share, multiples);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(density_prior_box,
                       ops::DensityPriorBoxOpNPUKernel<plat::float16>,
                       ops::DensityPriorBoxOpNPUKernel<float>);
