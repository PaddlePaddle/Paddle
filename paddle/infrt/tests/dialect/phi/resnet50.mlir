// RUN: infrtexec -i %s
module  {
  func @main_graph(%arg0: !phi.dense_tensor_map, %arg1: !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW> {
    %0 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %1 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %2 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_6.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %3 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %4 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %5 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %6 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %7 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %8 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %9 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %10 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %11 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_2.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %12 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %13 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %14 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %15 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_29.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %16 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_35.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %17 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %18 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %19 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %20 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_22.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %21 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_27.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %22 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %23 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %24 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_37.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %25 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_18.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %26 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_38.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %27 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_39.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %28 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_43.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %29 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_3.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %30 = phi_dt.tensor_map_get_tensor(%arg0) {name = "linear_0.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %31 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %32 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_49.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %33 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %34 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_8.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %35 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %36 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %37 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %38 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %39 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %40 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %41 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %42 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %43 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %44 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_7.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %45 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %46 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_40.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %47 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %48 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %49 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %50 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %51 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %52 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %53 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %54 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %55 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %56 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %57 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %58 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %59 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %60 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %61 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %62 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %63 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %64 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_47.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %65 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %66 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %67 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %68 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %69 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %70 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %71 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %72 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %73 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %74 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_11.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %75 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_30.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %76 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %77 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %78 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %79 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %80 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %81 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %82 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %83 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %84 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_13.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %85 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %86 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_0.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %87 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %88 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %89 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %90 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %91 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %92 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %93 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %94 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %95 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %96 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %97 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %98 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %99 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_4.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %100 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %101 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %102 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %103 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %104 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_44.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %105 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_24.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %106 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %107 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %108 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_20.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %109 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %110 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %111 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_23.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %112 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %113 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_25.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %114 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %115 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %116 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %117 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %118 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %119 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %120 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %121 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %122 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %123 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %124 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_34.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %125 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %126 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %127 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %128 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %129 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %130 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %131 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %132 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_52.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %133 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %134 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_9.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %135 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %136 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %137 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %138 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %139 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %140 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %141 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %142 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %143 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %144 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %145 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %146 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %147 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %148 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %149 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %150 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %151 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %152 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %153 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %154 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_1.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %155 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_21.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %156 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %157 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_28.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %158 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_31.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %159 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %160 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_19.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %161 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %162 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %163 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %164 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %165 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %166 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %167 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %168 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %169 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %170 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %171 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %172 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %173 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %174 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %175 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_15.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %176 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %177 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %178 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_42.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %179 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %180 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %181 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %182 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %183 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %184 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %185 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %186 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %187 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %188 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %189 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %190 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_50.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %191 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %192 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %193 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %194 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %195 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %196 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %197 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_12.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %198 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %199 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_16.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %200 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %201 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %202 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %203 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_48.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %204 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %205 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %206 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %207 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %208 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %209 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %210 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %211 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_36.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %212 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %213 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %214 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_32.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %215 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %216 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %217 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %218 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %219 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_10.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %220 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %221 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %222 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %223 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %224 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %225 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %226 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %227 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %228 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %229 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %230 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %231 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_5.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %232 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %233 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_33.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %234 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_46.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %235 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_45.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %236 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %237 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %238 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %239 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_14.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %240 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %241 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %242 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_26.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %243 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %244 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %245 = phi_dt.tensor_map_get_tensor(%arg0) {name = "linear_0.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %246 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_51.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %247 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_41.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %248 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %249 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %250 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %251 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %252 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %253 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %254 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %255 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %256 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %257 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.w_1"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %258 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %259 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %260 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %261 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %262 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %263 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_17.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %264 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.w_2"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %265 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %266 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %267 = "pd.conv2d"(%arg1, %86) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [3 : i32, 3 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y, %MeanOut, %VarianceOut = "pd.batch_norm"(%267, %259, %189, %115, %81) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %268 = "pd.relu"(%Y) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %269 = "pd.pool2d"(%268) {adaptive = false, ceil_mode = false, data_format = "NCHW", exclusive = true, global_pooling = false, ksize = [3 : i32, 3 : i32], padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], pooling_type = "max", strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %270 = "pd.conv2d"(%269, %11) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_0, %MeanOut_1, %VarianceOut_2 = "pd.batch_norm"(%270, %150, %107, %144, %202) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %271 = "pd.relu"(%Y_0) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %272 = "pd.conv2d"(%271, %29) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_3, %MeanOut_4, %VarianceOut_5 = "pd.batch_norm"(%272, %205, %159, %89, %114) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %273 = "pd.relu"(%Y_3) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %274 = "pd.conv2d"(%273, %99) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_6, %MeanOut_7, %VarianceOut_8 = "pd.batch_norm"(%274, %168, %135, %257, %230) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %275 = "pd.conv2d"(%269, %154) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_9, %MeanOut_10, %VarianceOut_11 = "pd.batch_norm"(%275, %79, %131, %69, %83) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %276 = "pd.elementwise_add"(%Y_6, %Y_9) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %277 = "pd.relu"(%276) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %278 = "pd.conv2d"(%277, %231) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_12, %MeanOut_13, %VarianceOut_14 = "pd.batch_norm"(%278, %37, %261, %171, %4) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %279 = "pd.relu"(%Y_12) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %280 = "pd.conv2d"(%279, %2) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_15, %MeanOut_16, %VarianceOut_17 = "pd.batch_norm"(%280, %206, %218, %236, %6) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %281 = "pd.relu"(%Y_15) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %282 = "pd.conv2d"(%281, %44) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_18, %MeanOut_19, %VarianceOut_20 = "pd.batch_norm"(%282, %162, %50, %43, %213) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %283 = "pd.elementwise_add"(%Y_18, %277) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %284 = "pd.relu"(%283) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %285 = "pd.conv2d"(%284, %34) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_21, %MeanOut_22, %VarianceOut_23 = "pd.batch_norm"(%285, %97, %14, %225, %138) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %286 = "pd.relu"(%Y_21) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %287 = "pd.conv2d"(%286, %134) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_24, %MeanOut_25, %VarianceOut_26 = "pd.batch_norm"(%287, %222, %198, %42, %136) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %288 = "pd.relu"(%Y_24) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %289 = "pd.conv2d"(%288, %219) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_27, %MeanOut_28, %VarianceOut_29 = "pd.batch_norm"(%289, %40, %194, %191, %82) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %290 = "pd.elementwise_add"(%Y_27, %284) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %291 = "pd.relu"(%290) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %292 = "pd.conv2d"(%291, %197) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_30, %MeanOut_31, %VarianceOut_32 = "pd.batch_norm"(%292, %229, %85, %51, %186) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %293 = "pd.relu"(%Y_30) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %294 = "pd.conv2d"(%293, %84) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_33, %MeanOut_34, %VarianceOut_35 = "pd.batch_norm"(%294, %3, %54, %88, %226) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %295 = "pd.relu"(%Y_33) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %296 = "pd.conv2d"(%295, %239) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_36, %MeanOut_37, %VarianceOut_38 = "pd.batch_norm"(%296, %196, %121, %204, %195) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %297 = "pd.conv2d"(%291, %74) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_39, %MeanOut_40, %VarianceOut_41 = "pd.batch_norm"(%297, %254, %262, %8, %106) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %298 = "pd.elementwise_add"(%Y_36, %Y_39) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %299 = "pd.relu"(%298) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %300 = "pd.conv2d"(%299, %175) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_42, %MeanOut_43, %VarianceOut_44 = "pd.batch_norm"(%300, %151, %100, %13, %109) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %301 = "pd.relu"(%Y_42) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %302 = "pd.conv2d"(%301, %199) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_45, %MeanOut_46, %VarianceOut_47 = "pd.batch_norm"(%302, %72, %77, %210, %101) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %303 = "pd.relu"(%Y_45) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %304 = "pd.conv2d"(%303, %263) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_48, %MeanOut_49, %VarianceOut_50 = "pd.batch_norm"(%304, %129, %149, %170, %112) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %305 = "pd.elementwise_add"(%Y_48, %299) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %306 = "pd.relu"(%305) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %307 = "pd.conv2d"(%306, %25) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_51, %MeanOut_52, %VarianceOut_53 = "pd.batch_norm"(%307, %98, %176, %153, %216) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %308 = "pd.relu"(%Y_51) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %309 = "pd.conv2d"(%308, %160) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_54, %MeanOut_55, %VarianceOut_56 = "pd.batch_norm"(%309, %145, %130, %192, %142) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %310 = "pd.relu"(%Y_54) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %311 = "pd.conv2d"(%310, %108) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_57, %MeanOut_58, %VarianceOut_59 = "pd.batch_norm"(%311, %117, %207, %174, %71) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %312 = "pd.elementwise_add"(%Y_57, %306) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %313 = "pd.relu"(%312) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %314 = "pd.conv2d"(%313, %155) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_60, %MeanOut_61, %VarianceOut_62 = "pd.batch_norm"(%314, %240, %12, %133, %128) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %315 = "pd.relu"(%Y_60) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %316 = "pd.conv2d"(%315, %20) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_63, %MeanOut_64, %VarianceOut_65 = "pd.batch_norm"(%316, %184, %180, %68, %127) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %317 = "pd.relu"(%Y_63) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %318 = "pd.conv2d"(%317, %111) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_66, %MeanOut_67, %VarianceOut_68 = "pd.batch_norm"(%318, %152, %243, %73, %146) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %319 = "pd.elementwise_add"(%Y_66, %313) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %320 = "pd.relu"(%319) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %321 = "pd.conv2d"(%320, %113) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_69, %MeanOut_70, %VarianceOut_71 = "pd.batch_norm"(%321, %179, %93, %126, %165) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %322 = "pd.relu"(%Y_69) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %323 = "pd.conv2d"(%322, %242) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_72, %MeanOut_73, %VarianceOut_74 = "pd.batch_norm"(%323, %17, %91, %185, %167) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %324 = "pd.relu"(%Y_72) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %325 = "pd.conv2d"(%324, %21) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_75, %MeanOut_76, %VarianceOut_77 = "pd.batch_norm"(%325, %143, %63, %7, %173) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %326 = "pd.conv2d"(%320, %105) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_78, %MeanOut_79, %VarianceOut_80 = "pd.batch_norm"(%326, %182, %266, %251, %67) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %327 = "pd.elementwise_add"(%Y_75, %Y_78) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %328 = "pd.relu"(%327) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %329 = "pd.conv2d"(%328, %157) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_81, %MeanOut_82, %VarianceOut_83 = "pd.batch_norm"(%329, %241, %22, %61, %172) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %330 = "pd.relu"(%Y_81) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %331 = "pd.conv2d"(%330, %15) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_84, %MeanOut_85, %VarianceOut_86 = "pd.batch_norm"(%331, %60, %200, %57, %38) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %332 = "pd.relu"(%Y_84) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %333 = "pd.conv2d"(%332, %75) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_87, %MeanOut_88, %VarianceOut_89 = "pd.batch_norm"(%333, %65, %87, %53, %95) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %334 = "pd.elementwise_add"(%Y_87, %328) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %335 = "pd.relu"(%334) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %336 = "pd.conv2d"(%335, %158) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_90, %MeanOut_91, %VarianceOut_92 = "pd.batch_norm"(%336, %80, %19, %49, %252) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %337 = "pd.relu"(%Y_90) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %338 = "pd.conv2d"(%337, %214) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_93, %MeanOut_94, %VarianceOut_95 = "pd.batch_norm"(%338, %70, %166, %102, %147) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %339 = "pd.relu"(%Y_93) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %340 = "pd.conv2d"(%339, %233) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_96, %MeanOut_97, %VarianceOut_98 = "pd.batch_norm"(%340, %66, %94, %39, %163) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %341 = "pd.elementwise_add"(%Y_96, %335) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %342 = "pd.relu"(%341) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %343 = "pd.conv2d"(%342, %124) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_99, %MeanOut_100, %VarianceOut_101 = "pd.batch_norm"(%343, %256, %265, %31, %209) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %344 = "pd.relu"(%Y_99) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %345 = "pd.conv2d"(%344, %16) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_102, %MeanOut_103, %VarianceOut_104 = "pd.batch_norm"(%345, %139, %248, %96, %181) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %346 = "pd.relu"(%Y_102) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %347 = "pd.conv2d"(%346, %211) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_105, %MeanOut_106, %VarianceOut_107 = "pd.batch_norm"(%347, %260, %78, %56, %58) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %348 = "pd.elementwise_add"(%Y_105, %342) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %349 = "pd.relu"(%348) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %350 = "pd.conv2d"(%349, %24) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_108, %MeanOut_109, %VarianceOut_110 = "pd.batch_norm"(%350, %76, %156, %238, %0) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %351 = "pd.relu"(%Y_108) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %352 = "pd.conv2d"(%351, %26) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_111, %MeanOut_112, %VarianceOut_113 = "pd.batch_norm"(%352, %10, %161, %116, %188) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %353 = "pd.relu"(%Y_111) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %354 = "pd.conv2d"(%353, %27) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_114, %MeanOut_115, %VarianceOut_116 = "pd.batch_norm"(%354, %52, %208, %140, %264) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %355 = "pd.elementwise_add"(%Y_114, %349) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %356 = "pd.relu"(%355) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %357 = "pd.conv2d"(%356, %46) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_117, %MeanOut_118, %VarianceOut_119 = "pd.batch_norm"(%357, %258, %9, %5, %169) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %358 = "pd.relu"(%Y_117) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %359 = "pd.conv2d"(%358, %247) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_120, %MeanOut_121, %VarianceOut_122 = "pd.batch_norm"(%359, %177, %255, %253, %193) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %360 = "pd.relu"(%Y_120) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %361 = "pd.conv2d"(%360, %178) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_123, %MeanOut_124, %VarianceOut_125 = "pd.batch_norm"(%361, %47, %201, %48, %249) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %362 = "pd.elementwise_add"(%Y_123, %356) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %363 = "pd.relu"(%362) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %364 = "pd.conv2d"(%363, %104) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_126, %MeanOut_127, %VarianceOut_128 = "pd.batch_norm"(%364, %164, %217, %110, %141) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %365 = "pd.relu"(%Y_126) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %366 = "pd.conv2d"(%365, %235) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_129, %MeanOut_130, %VarianceOut_131 = "pd.batch_norm"(%366, %137, %125, %224, %35) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %367 = "pd.relu"(%Y_129) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %368 = "pd.conv2d"(%367, %234) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_132, %MeanOut_133, %VarianceOut_134 = "pd.batch_norm"(%368, %227, %55, %123, %23) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %369 = "pd.conv2d"(%363, %28) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_135, %MeanOut_136, %VarianceOut_137 = "pd.batch_norm"(%369, %187, %36, %41, %250) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %370 = "pd.elementwise_add"(%Y_132, %Y_135) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %371 = "pd.relu"(%370) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %372 = "pd.conv2d"(%371, %64) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_138, %MeanOut_139, %VarianceOut_140 = "pd.batch_norm"(%372, %1, %122, %118, %220) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %373 = "pd.relu"(%Y_138) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %374 = "pd.conv2d"(%373, %203) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_141, %MeanOut_142, %VarianceOut_143 = "pd.batch_norm"(%374, %237, %120, %212, %92) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %375 = "pd.relu"(%Y_141) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %376 = "pd.conv2d"(%375, %32) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_144, %MeanOut_145, %VarianceOut_146 = "pd.batch_norm"(%376, %244, %59, %183, %228) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %377 = "pd.elementwise_add"(%Y_144, %371) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %378 = "pd.relu"(%377) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %379 = "pd.conv2d"(%378, %190) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_147, %MeanOut_148, %VarianceOut_149 = "pd.batch_norm"(%379, %119, %103, %18, %45) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %380 = "pd.relu"(%Y_147) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %381 = "pd.conv2d"(%380, %246) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_150, %MeanOut_151, %VarianceOut_152 = "pd.batch_norm"(%381, %148, %232, %221, %62) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %382 = "pd.relu"(%Y_150) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %383 = "pd.conv2d"(%382, %132) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y_153, %MeanOut_154, %VarianceOut_155 = "pd.batch_norm"(%383, %215, %90, %33, %223) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %384 = "pd.elementwise_add"(%Y_153, %378) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %385 = "pd.relu"(%384) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %386 = "pd.pool2d"(%385) {adaptive = true, ceil_mode = false, data_format = "NCHW", exclusive = true, global_pooling = false, ksize = [1 : i32, 1 : i32], padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], pooling_type = "avg", strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %387 = "pd.flatten_contiguous_range"(%386) {start_axis = 1 : si32, stop_axis = 3 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %388 = "pd.matmul_v2"(%387, %245) {trans_x = false, trans_y = false} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %389 = "pd.elementwise_add"(%388, %30) {axis = 1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    infrt.return %270 : !infrt.dense_tensor<CPU, FP32, NCHW>
  }

  func @main() {
    %ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>
    %1 = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value = 12.0 : f32, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[1, 3, 256, 256]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %map = phi_dt.load_combined_params(){model_path="/shixiaowei02/Paddle-InfRT/Paddle/build_rel_debug_info/models/resnet50/model.pdmodel",params_path="/shixiaowei02/Paddle-InfRT/Paddle/build_rel_debug_info/models/resnet50/model.pdiparams"}
    %2 = infrt.call@main_graph(%map, %1) : (!phi.dense_tensor_map, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    phi_dt.print_tensor (%2 : !infrt.dense_tensor<CPU, FP32, NCHW>)
    infrt.return
  }
  }
