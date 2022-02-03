#include "paddle/pten/kernels/funcs/eigen/common.h"
#include "paddle/pten/kernels/batch_norm_kernel.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/pten/kernels/gpu/batch_norm_utils.h"

namespace pten {


template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;


template <typename T, typename Context>
void BatchNormGradGradKernel(const Context& ctx,  
                    const DenseTensor& x_grad_grad, const DenseTensor& scale_grad_grad,
                    const DenseTensor& bias_grad_grad, const DenseTensor& y_grad,
                    const DenseTensor& x, \
                    const DenseTensor& scale, const DenseTensor& bias,
                    const DenseTensor& saved_mean, const DenseTensor& saved_variance,
                    paddle::optional<const DenseTensor&> mean,
                    paddle::optional<const DenseTensor&> variance,                    
                    float momentum, float epsilon, const std::string& data_layout_str,
                    bool is_test, bool use_global_stats, bool trainable_statistics,
                    bool fuse_with_relu,
                    DenseTensor* x_grad, DenseTensor* scale_grad, DenseTensor* y_grad_grad ){
    const auto *X = &x;
    const auto *Scale = &scale;
    const auto *dY = &y_grad;
    const auto *Saved_mean = &saved_mean;
    const auto *Saved_variance = &saved_variance;

    PADDLE_ENFORCE_EQ(
        is_test, false,
        platform::errors::InvalidArgument(
            "`is_test = True` CANNOT be used in train program. If "
            "you want to use global status in pre_train model, "
            "please set `use_global_stats = True`"));

    const auto data_layout =
        paddle::framework::StringToDataLayout(data_layout_str);

    const auto *ddX = &x_grad_grad;
    const auto *ddScale = &scale_grad_grad;
    const auto *ddBias = &bias_grad_grad;

    auto *dX = x_grad;
    auto *dScale = scale_grad;
    auto *ddY = y_grad_grad;
    dX->mutable_data<T>(ctx.GetPlace());
    ddY->mutable_data<T>(ctx.GetPlace());

    const auto &x_dims = X->dims();
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int sample_size = X->numel() / C;
    paddle::operators::math::SetConstant<Context, T> set_constant;

    const T *mean_data = Saved_mean->data<T>();
    const T *inv_var_data = Saved_variance->data<T>();

    DenseTensor inv_var_tensor;
    if (use_global_stats) {
      const auto *running_mean = mean.get_ptr();
      const auto *running_variance = variance.get_ptr();
      mean_data = running_mean->data<T>();
      inv_var_tensor.Resize({C});

      T *running_inv_var_data = inv_var_tensor.mutable_data<T>(ctx.GetPlace());
      EigenVectorArrayMap<T> inv_var_tmp(running_inv_var_data, C);
      ConstEigenVectorArrayMap<T> var_arr(running_variance->data<T>(), C);

      inv_var_tmp = (var_arr + epsilon).sqrt().inverse();
      inv_var_data = running_inv_var_data;
    }

    // transpose NCHW -> NHWC for easy calculate
    DenseTensor transformed_x(X->type());
    DenseTensor transformed_dy(dY->type());
    DenseTensor transformed_ddx(ddX->type());

    DenseTensor transformed_dx(dX->type());
    DenseTensor transformed_ddy(ddY->type());
    if (data_layout == DataLayout::kNCHW && x_dims.size() > 2) {
      VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
      // Input Tensor
      ResizeToChannelLast<Context, T>(ctx, X,
                                                         &transformed_x);
      TransToChannelLast<Context, T>(ctx, X, &transformed_x);
      ResizeToChannelLast<Context, T>(ctx, dY,
                                                         &transformed_dy);
      TransToChannelLast<Context, T>(ctx, dY,
                                                        &transformed_dy);
      ResizeToChannelLast<Context, T>(ctx, ddX,
                                                         &transformed_ddx);
      TransToChannelLast<Context, T>(ctx, ddX,
                                                        &transformed_ddx);
      // Output Tensor
      ResizeToChannelLast<Context, T>(ctx, dX,
                                                         &transformed_dx);
      ResizeToChannelLast<Context, T>(ctx, ddY,
                                                         &transformed_ddy);
    } else {
      transformed_x.ShareDataWith(*X);
      transformed_dy.ShareDataWith(*dY);
      transformed_ddx.ShareDataWith(*ddX);

      transformed_dx.ShareDataWith(*dX);
      transformed_ddy.ShareDataWith(*ddY);
    }

    ConstEigenArrayMap<T> x_arr(transformed_x.data<T>(), C, sample_size);
    ConstEigenVectorArrayMap<T> mean_arr(mean_data, C);
    ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, C);

    Tensor mean_tile;
    mean_tile.Resize({C, sample_size});
    mean_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> mean_tile_data(mean_tile.mutable_data<T>(ctx.GetPlace()),
                                    C, sample_size);

    DenseTensor inv_var_tile;
    inv_var_tile.Resize({C, sample_size});
    inv_var_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> inv_var_tile_data(
        inv_var_tile.mutable_data<T>(ctx.GetPlace()), C, sample_size);

    mean_tile_data = mean_arr.replicate(1, sample_size);
    inv_var_tile_data = inv_var_arr.replicate(1, sample_size);

    DenseTensor Scale_data;
    if (!Scale) {
      Scale_data.mutable_data<T>({C}, ctx.GetPlace());
      set_constant(ctx, &Scale_data, static_cast<T>(1));
    }
    ConstEigenVectorArrayMap<T> scale_arr(
        Scale ? Scale->data<T>() : Scale_data.data<T>(), C);

    Tensor scale_tile;
    scale_tile.Resize({C, sample_size});
    scale_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> scale_tile_data(scale_tile.mutable_data<T>(ctx.GetPlace()),
                                     C, sample_size);
    scale_tile_data = scale_arr.replicate(1, sample_size);

    ConstEigenArrayMap<T> dy_arr(transformed_dy.data<T>(), C, sample_size);
    ConstEigenArrayMap<T> ddx_arr(transformed_ddx.data<T>(), C, sample_size);

    DenseTensor x_sub_mean_mul_invstd;
    x_sub_mean_mul_invstd.Resize({C, sample_size});
    x_sub_mean_mul_invstd.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> x_sub_mean_mul_invstd_arr(
        x_sub_mean_mul_invstd.mutable_data<T>(ctx.GetPlace()), C, sample_size);
    x_sub_mean_mul_invstd_arr = (x_arr - mean_tile_data) * inv_var_tile_data;

    if (dX) {
      dX->mutable_data<T>(ctx.GetPlace());
      EigenArrayMap<T> dx_arr(transformed_dx.mutable_data<T>(ctx.GetPlace()), C,
                              sample_size);
      dx_arr.setZero();
      if (use_global_stats) {
        // math: dx = (ddscale * dy) * inv_var
        if (ddScale) {
          ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
          Tensor ddscale_tile;
          ddscale_tile.Resize({C, sample_size});
          EigenArrayMap<T> ddscale_tile_data(
              ddscale_tile.mutable_data<T>(ctx.GetPlace()), C, sample_size);
          ddscale_tile_data = ddscale_arr.replicate(1, sample_size);

          dx_arr = dy_arr * ddscale_tile_data * inv_var_tile_data;
        }
      } else {
        // math: dx = scale * ((x - mean) * inv_var / NxHxW * (np.mean(ddx,
        // axis=(n,h,w)) *
        //          np.sum(dy, axis=(n,h,w)) -
        //          np.sum(dy * ddx, axis=(n,h,w)) + 3 * np.mean(dy * (x -
        //          mean),
        //          axis=(n,h,w)) * inv_var.pow(2) *
        //          np.sum(ddx * (x - mean), axis=(n,h,w))) + inv_var.pow(3) /
        //          NxHxW *
        //          np.sum(ddx * (x - mean)) *
        //          (np.mean(dy, axis=(n,h,w)) - dy) + inv_var.pow(3) / NxHxW *
        //          np.sum(dy,
        //          axis=(n,h,w)) * (x - mean) *
        //          (np.mean(ddx, axis=(n,h,w)) - ddx)) + ddr * (dy * inv_var -
        //          inv_var
        //          *
        //          np.mean(dy, axis=(n,h,w)) -
        //          inv_var.pow(3) * (x - mean) * np.mean(dy * (x - mean),
        //          axis=(n,h,w)))

        if (ddX) {
          dx_arr +=
              (x_sub_mean_mul_invstd_arr * inv_var_tile_data *
               inv_var_tile_data / sample_size)
                  .colwise() *
              (ddx_arr.rowwise().sum() * dy_arr.rowwise().sum() / sample_size -
               (dy_arr * ddx_arr).rowwise().sum() +
               3. * (dy_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() *
                   (ddx_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() /
                   sample_size);

          dx_arr += (inv_var_tile_data * inv_var_tile_data).colwise() *
                    (ddx_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() /
                    sample_size *
                    (dy_arr.rowwise().sum() / sample_size - dy_arr);

          dx_arr += (inv_var_tile_data * inv_var_tile_data).colwise() *
                    (dy_arr * x_sub_mean_mul_invstd_arr).rowwise().sum() /
                    sample_size *
                    (ddx_arr.rowwise().sum() / sample_size - ddx_arr);

          dx_arr = scale_tile_data * dx_arr;
        }
        if (ddScale) {
          ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
          Tensor ddscale_tile;
          ddscale_tile.Resize({C, sample_size});
          EigenArrayMap<T> ddscale_tile_data(
              ddscale_tile.mutable_data<T>(ctx.GetPlace()), C, sample_size);
          ddscale_tile_data = ddscale_arr.replicate(1, sample_size);

          dx_arr += (dy_arr * inv_var_tile_data -
                     (dy_arr.rowwise().sum().replicate(1, sample_size) /
                      sample_size) *
                         inv_var_tile_data -
                     x_sub_mean_mul_invstd_arr * inv_var_tile_data *
                         (dy_arr * x_sub_mean_mul_invstd_arr)
                             .rowwise()
                             .sum()
                             .replicate(1, sample_size) /
                         sample_size) *
                    ddscale_tile_data;
        }
      }
      if (data_layout == DataLayout::kNCHW) {
        VLOG(3) << "Transform batchnorm output from NHWC to NCHW";
        TransToChannelFirst<Context, T>(
            ctx, &transformed_dx, dX);
      }
    }
    if (dScale) {
      dScale->mutable_data<T>(ctx.GetPlace());
      EigenVectorArrayMap<T> dscale_arr(dScale->mutable_data<T>(ctx.GetPlace()),
                                        C);
      dscale_arr.setZero();
      if (use_global_stats) {
        // math: dscale = np.sum(ddx * dy, axis=(n,h,w)) * inv_var
        if (ddX) {
          dscale_arr = (ddx_arr * dy_arr * inv_var_tile_data).rowwise().sum();
        }
      } else {
        // math: dscale = inv_var * (dy - np.mean(dy, axis=(n,h,w) - (x-mean) *
        //            inv_var.pow(2) * np.mean(dy * (x-mean), axis=(n,h,w)))) *
        //            ddx
        if (ddX) {
          Tensor first_grad;
          first_grad.Resize({C, sample_size});
          EigenArrayMap<T> first_grad_arr(
              first_grad.mutable_data<T>(ctx.GetPlace()), C, sample_size);
          first_grad_arr.setZero();

          first_grad_arr +=
              inv_var_tile_data *
              (dy_arr -
               dy_arr.rowwise().sum().replicate(1, sample_size) / sample_size -
               x_sub_mean_mul_invstd_arr *
                   (dy_arr * x_sub_mean_mul_invstd_arr)
                       .rowwise()
                       .sum()
                       .replicate(1, sample_size) /
                   sample_size);
          dscale_arr = (first_grad_arr * ddx_arr).rowwise().sum();
        }
      }
    }

    if (ddY) {
      ddY->mutable_data<T>(ctx.GetPlace());
      EigenArrayMap<T> ddy_arr(transformed_ddy.mutable_data<T>(ctx.GetPlace()),
                               C, sample_size);
      ddy_arr.setZero();
      if (use_global_stats) {
        // math: ddy = r * ddx * inv_var + ddbias +
        //           ddscale * (x - mean) * inv_var
        if (ddX) {
          ddy_arr = scale_tile_data * ddx_arr * inv_var_tile_data;
        }
      } else {
        // math: ddy = (x - mean) * inv_var * ddscale + ddbias +
        //           scale * inv_var * (ddx - (x - mean) * inv_var.pow(2) *
        //           np.mean(ddx * (x - mean), axis=(n,h,w)))
        if (ddX) {
          ddy_arr +=
              scale_tile_data * inv_var_tile_data *
              (ddx_arr -
               ddx_arr.rowwise().sum().replicate(1, sample_size) / sample_size -
               x_sub_mean_mul_invstd_arr *
                   (ddx_arr * x_sub_mean_mul_invstd_arr)
                       .rowwise()
                       .sum()
                       .replicate(1, sample_size) /
                   sample_size);
        }
      }
      if (ddScale) {
        ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
        Tensor ddscale_tile;
        ddscale_tile.Resize({C, sample_size});
        EigenArrayMap<T> ddscale_tile_data(
            ddscale_tile.mutable_data<T>(ctx.GetPlace()), C, sample_size);
        ddscale_tile_data = ddscale_arr.replicate(1, sample_size);

        ddy_arr += x_sub_mean_mul_invstd_arr * ddscale_tile_data;
      }

      if (ddBias) {
        ConstEigenVectorArrayMap<T> ddbias_arr(ddBias->data<T>(), C);
        Tensor ddbias_tile;
        ddbias_tile.Resize({C, sample_size});
        EigenArrayMap<T> ddbias_tile_data(
            ddbias_tile.mutable_data<T>(ctx.GetPlace()), C, sample_size);
        ddbias_tile_data = ddbias_arr.replicate(1, sample_size);

        ddy_arr += ddbias_tile_data;
      }

      if (data_layout == DataLayout::kNCHW) {
        VLOG(3) << "Transform batchnorm output from NHWC to NCHW";
        TransToChannelFirst<Context, T>(
            ctx, &transformed_ddy, ddY);
      }
    }

}

} // namespace pten


PT_REGISTER_KERNEL(batch_norm_grad_grad, CPU, ALL_LAYOUT, pten::BatchNormGradGradKernel, float, double) {}
