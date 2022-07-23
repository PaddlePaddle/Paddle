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

namespace phi {

template <typename T, typename Context>
void LUGradKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& out,
                const DenseTensor& pivots,
                const DenseTensor& out_grad,
                bool pivot,
                DenseTensor* x_grad){
    auto xin = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Input<framework::Tensor>("Out");
    auto P = ctx.Input<framework::Tensor>("Pivots");
    auto dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    const auto& dev_ctx = ctx.template device_context<DeviceContext>();
    math::DeviceIndependenceTensorOperations<DeviceContext, T> helper(ctx);
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);

    auto xdims = xin->dims();
    int xrank = xdims.size();
    int64_t m = xdims[xrank - 2];
    int64_t n = xdims[xrank - 1];
    int64_t k = std::min(m, n);

    framework::Tensor L, U, L_narrow, U_narrow, L_narrow_mH, U_narrow_mH,
        grad_narrow;
    LU_Unpack<DeviceContext, T>(dev_ctx, out, &L, &U);

    Tensor_narrow<DeviceContext, T>(ctx, &L, &L_narrow, 0, k, 0, k);
    Tensor_narrow<DeviceContext, T>(ctx, &U, &U_narrow, 0, k, 0, k);
    Tensor_narrow<DeviceContext, T>(ctx, dout, &grad_narrow, 0, k, 0, k);
    auto graddims = grad_narrow.dims();

    Tensor_Conj<DeviceContext, T>(dev_ctx, L_narrow, &L_narrow_mH);
    Tensor_Conj<DeviceContext, T>(dev_ctx, U_narrow, &U_narrow_mH);
    L_narrow_mH = helper.Transpose(L_narrow_mH);
    U_narrow_mH = helper.Transpose(U_narrow_mH);

    auto LmHdims = L_narrow_mH.dims();
    auto UmHdims = U_narrow_mH.dims();

    framework::Tensor phi_L, phi_U, phi, psi;
    phi_L.Resize(LmHdims);
    phi_L.mutable_data<T>(ctx.GetPlace());
    phi_U.Resize(UmHdims);
    phi_U.mutable_data<T>(ctx.GetPlace());
    auto mat_dim_l = phi::funcs::CreateMatrixDescriptor(LmHdims, 0, false);
    auto mat_dim_u = phi::funcs::CreateMatrixDescriptor(UmHdims, 0, false);
    auto mat_dim_g = phi::funcs::CreateMatrixDescriptor(graddims, 0, false);
    blas.MatMul(L_narrow_mH,
                mat_dim_l,
                grad_narrow,
                mat_dim_g,
                static_cast<T>(1),
                &phi_L,
                static_cast<T>(0));

    blas.MatMul(grad_narrow,
                mat_dim_g,
                U_narrow_mH,
                mat_dim_u,
                static_cast<T>(1),
                &phi_U,
                static_cast<T>(0));

    auto phil_rank = LmHdims.size();
    auto phiu_rank = UmHdims.size();
    platform::ForRange<DeviceContext> l_for_range(dev_ctx, phi_L.numel());
    phi::funcs::TrilTriuCompute<T> tril_computer(phi_L.data<T>(),
                                                    -1,
                                                    true,
                                                    LmHdims[phil_rank - 2],
                                                    LmHdims[phil_rank - 1],
                                                    phi_L.data<T>());
    l_for_range(tril_computer);

    platform::ForRange<DeviceContext> u_for_range(dev_ctx, phi_U.numel());
    phi::funcs::TrilTriuCompute<T> triu_computer(phi_U.data<T>(),
                                                    0,
                                                    false,
                                                    UmHdims[phiu_rank - 2],
                                                    UmHdims[phiu_rank - 1],
                                                    phi_U.data<T>());
    u_for_range(triu_computer);

    Tensor_Add<DeviceContext, T>(dev_ctx, phi_L, phi_U, &phi);
    psi.Resize(xdims);
    psi.mutable_data<T>(ctx.GetPlace());
    phi::funcs::SetConstant<DeviceContext, T> setter;
    setter(dev_ctx, &psi, static_cast<T>(0));

    std::vector<int64_t> axes = {xrank - 2, xrank - 1};
    std::vector<int64_t> slice_starts(2, 0);
    std::vector<int64_t> slice_ends(2, 0);
    auto valuedims = vectorize(xdims);

    framework::Tensor Pmat;
    Unpack_Pivot<DeviceContext, T>(dev_ctx, *P, &Pmat, m, k);

    using Context =
        typename framework::ConvertToPhiContext<DeviceContext>::TYPE;
    auto& phi_dev_ctx = static_cast<const Context&>(dev_ctx);

    if (m <= n) {
        if (k < n) {
        framework::Tensor U_complement, U_grad_complement, phi_complement,
            phi_complement_l;
        Tensor_narrow<DeviceContext, T>(ctx, &U, &U_complement, 0, k, k, n);
        Tensor_narrow<DeviceContext, T>(
            ctx, dout, &U_grad_complement, 0, k, k, n);
        framework::Tensor U_complement_mH = helper.Transpose(U_complement);

        Tensor_Conj<DeviceContext, T>(
            dev_ctx, U_complement_mH, &U_complement_mH);

        auto mat_dim_g = phi::funcs::CreateMatrixDescriptor(
            U_grad_complement.dims(), 0, false);
        auto mat_dim_u = phi::funcs::CreateMatrixDescriptor(
            U_complement_mH.dims(), 0, false);
        auto phidims = UmHdims;
        phidims[UmHdims.size() - 2] = k;
        phidims[UmHdims.size() - 1] = k;
        phi_complement.Resize(phidims);
        phi_complement.mutable_data<T>(ctx.GetPlace());
        blas.MatMul(U_grad_complement,
                    mat_dim_g,
                    U_complement_mH,
                    mat_dim_u,
                    static_cast<T>(1),
                    &phi_complement,
                    static_cast<T>(0));

        phi_complement_l.Resize(phidims);
        phi_complement_l.mutable_data<T>(ctx.GetPlace());
        const auto H = phidims[phidims.size() - 2];
        const auto W = phidims[phidims.size() - 1];
        platform::ForRange<DeviceContext> x_for_range(dev_ctx,
                                                        phi_complement.numel());
        phi::funcs::TrilTriuCompute<T> tril_computer(
            phi_complement.data<T>(),
            -1,
            true,
            H,
            W,
            phi_complement_l.data<T>());
        x_for_range(tril_computer);

        Tensor_Sub<DeviceContext, T>(dev_ctx, phi, phi_complement_l, &phi);

        slice_starts[0] = 0;
        slice_starts[1] = k;
        slice_ends[0] = k;
        slice_ends[1] = n;
        valuedims[xrank - 2] = k;
        valuedims[xrank - 1] = n - k;
        SetValueCompute_dispatch<DeviceContext, T>(ctx,
                                                    &psi,
                                                    &U_grad_complement,
                                                    &psi,
                                                    axes,
                                                    &slice_starts,
                                                    &slice_ends,
                                                    valuedims,
                                                    xrank);
        }

        framework::Tensor psi_principal, phi_mH, psi_tmp;
        Tensor_Conj<DeviceContext, T>(dev_ctx, phi, &phi_mH);
        phi_mH = helper.Transpose(phi_mH);

        phi::TriangularSolveKernel<T, Context>(
            phi_dev_ctx, U_narrow, phi_mH, true, false, false, &psi_principal);

        Tensor_Conj<DeviceContext, T>(dev_ctx, psi_principal, &psi_principal);
        psi_principal = helper.Transpose(psi_principal);
        slice_starts[0] = 0;
        slice_starts[1] = 0;
        slice_ends[0] = k;
        slice_ends[1] = k;
        valuedims[xrank - 2] = k;
        valuedims[xrank - 1] = k;

        SetValueCompute_dispatch<DeviceContext, T>(ctx,
                                                    &psi,
                                                    &psi_principal,
                                                    &psi,
                                                    axes,
                                                    &slice_starts,
                                                    &slice_ends,
                                                    valuedims,
                                                    xrank);

        phi::TriangularSolveKernel<T, Context>(
            phi_dev_ctx, L_narrow_mH, psi, true, false, true, &psi_tmp);

        auto mat_dim_p =
            phi::funcs::CreateMatrixDescriptor(Pmat.dims(), 0, false);
        auto mat_dim_b =
            phi::funcs::CreateMatrixDescriptor(psi_tmp.dims(), 0, false);
        blas.MatMul(Pmat,
                    mat_dim_p,
                    psi_tmp,
                    mat_dim_b,
                    static_cast<T>(1),
                    dx,
                    static_cast<T>(0));
    } else {
        framework::Tensor L_complement, L_grad_complement, phi_complement,
            phi_complement_u;
        Tensor_narrow<DeviceContext, T>(ctx, &L, &L_complement, k, m, 0, k);
        Tensor_narrow<DeviceContext, T>(
            ctx, dout, &L_grad_complement, k, m, 0, k);
        framework::Tensor L_complement_mH = helper.Transpose(L_complement);
        Tensor_Conj<DeviceContext, T>(dev_ctx, L_complement_mH, &L_complement_mH);

        auto mat_dim_g = phi::funcs::CreateMatrixDescriptor(
            L_grad_complement.dims(), 0, false);
        auto mat_dim_u =
            phi::funcs::CreateMatrixDescriptor(L_complement_mH.dims(), 0, false);
        auto phidims = LmHdims;
        phidims[LmHdims.size() - 2] = k;
        phidims[LmHdims.size() - 1] = k;
        phi_complement.Resize(phidims);
        phi_complement.mutable_data<T>(ctx.GetPlace());
        blas.MatMul(L_complement_mH,
                    mat_dim_u,
                    L_grad_complement,
                    mat_dim_g,
                    static_cast<T>(1),
                    &phi_complement,
                    static_cast<T>(0));

        phi_complement_u.Resize(phidims);
        phi_complement_u.mutable_data<T>(ctx.GetPlace());
        const auto H = phidims[phidims.size() - 2];
        const auto W = phidims[phidims.size() - 1];
        platform::ForRange<DeviceContext> x_for_range(dev_ctx,
                                                    phi_complement.numel());
        phi::funcs::TrilTriuCompute<T> triu_computer(
            phi_complement.data<T>(), 0, false, H, W, phi_complement_u.data<T>());
        x_for_range(triu_computer);

        Tensor_Sub<DeviceContext, T>(dev_ctx, phi, phi_complement_u, &phi);

        slice_starts[0] = k;
        slice_starts[1] = 0;
        slice_ends[0] = m;
        slice_ends[1] = k;
        valuedims[xrank - 2] = m - k;
        valuedims[xrank - 1] = k;
        SetValueCompute_dispatch<DeviceContext, T>(ctx,
                                                    &psi,
                                                    &L_grad_complement,
                                                    &psi,
                                                    axes,
                                                    &slice_starts,
                                                    &slice_ends,
                                                    valuedims,
                                                    xrank);
        framework::Tensor psi_principal, phi_mH, psi_tmp, U_narrow_mH;

        phi::TriangularSolveKernel<T, Context>(
            phi_dev_ctx, L_narrow_mH, phi, true, false, true, &psi_principal);

        slice_starts[0] = 0;
        slice_starts[1] = 0;
        slice_ends[0] = k;
        slice_ends[1] = k;
        valuedims[xrank - 2] = k;
        valuedims[xrank - 1] = k;

        SetValueCompute_dispatch<DeviceContext, T>(ctx,
                                                    &psi,
                                                    &psi_principal,
                                                    &psi,
                                                    axes,
                                                    &slice_starts,
                                                    &slice_ends,
                                                    valuedims,
                                                    xrank);

        psi_tmp.Resize(psi.dims());
        psi_tmp.mutable_data<T>(ctx.GetPlace());
        auto mat_dim_p =
            phi::funcs::CreateMatrixDescriptor(Pmat.dims(), 0, false);
        auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(psi.dims(), 0, false);
        blas.MatMul(Pmat,
                    mat_dim_p,
                    psi,
                    mat_dim_b,
                    static_cast<T>(1),
                    &psi_tmp,
                    static_cast<T>(0));
        psi_tmp = helper.Transpose(psi_tmp);

        Tensor_Conj<DeviceContext, T>(dev_ctx, U_narrow, &U_narrow_mH);
        phi::TriangularSolveKernel<T, Context>(
            phi_dev_ctx, U_narrow_mH, psi_tmp, true, false, false, &psi);
        *dx = helper.Transpose(psi);
    }
}

}  // namespace phi