module  {
  func @main_graph(%arg0: !phi.dense_tensor_map, %arg1: !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW> {
    %0 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %1 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %2 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_6.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %3 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %4 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %5 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %6 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %7 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %8 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %9 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %10 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %11 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_2.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %12 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %13 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %14 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %15 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_29.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %16 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_35.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %17 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %18 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %19 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %20 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_22.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %21 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_27.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %22 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %23 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %24 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_37.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %25 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_18.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %26 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_38.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %27 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_39.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %28 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_43.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %29 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_3.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %30 = phi_dt.tensor_map_get_tensor(%arg0) {name = "linear_0.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %31 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %32 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_49.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %33 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %34 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_8.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %35 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %36 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %37 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %38 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %39 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %40 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %41 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %42 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %43 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %44 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_7.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %45 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %46 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_40.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %47 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %48 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %49 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %50 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %51 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %52 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %53 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %54 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %55 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %56 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %57 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %58 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %59 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %60 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %61 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %62 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %63 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %64 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_47.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %65 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %66 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %67 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %68 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %69 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %70 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %71 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %72 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %73 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %74 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_11.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %75 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_30.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %76 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %77 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %78 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %79 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %80 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %81 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %82 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %83 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %84 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_13.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %85 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %86 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_0.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %87 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %88 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %89 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %90 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %91 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %92 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %93 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %94 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %95 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %96 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %97 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %98 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %99 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_4.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %100 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %101 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %102 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %103 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %104 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_44.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %105 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_24.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %106 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %107 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %108 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_20.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %109 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %110 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %111 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_23.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %112 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %113 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_25.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %114 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %115 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %116 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %117 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %118 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %119 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %120 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %121 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %122 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %123 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %124 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_34.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %125 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %126 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %127 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %128 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %129 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %130 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %131 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %132 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_52.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %133 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %134 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_9.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %135 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %136 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %137 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %138 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %139 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %140 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %141 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %142 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %143 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %144 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %145 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %146 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %147 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %148 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %149 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %150 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %151 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %152 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %153 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %154 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_1.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %155 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_21.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %156 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %157 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_28.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %158 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_31.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %159 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %160 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_19.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %161 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %162 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %163 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %164 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %165 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %166 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %167 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %168 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %169 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %170 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %171 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %172 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %173 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %174 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %175 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_15.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %176 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %177 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %178 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_42.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %179 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %180 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %181 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %182 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %183 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %184 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %185 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %186 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %187 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %188 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %189 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %190 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_50.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %191 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %192 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %193 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %194 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %195 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %196 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %197 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_12.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %198 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %199 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_16.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %200 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %201 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %202 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %203 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_48.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %204 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %205 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %206 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %207 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %208 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %209 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %210 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %211 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_36.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %212 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %213 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %214 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_32.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %215 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %216 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %217 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %218 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %219 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_10.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %220 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %221 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %222 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %223 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %224 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %225 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %226 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %227 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %228 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %229 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %230 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %231 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_5.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %232 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %233 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_33.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %234 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_46.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %235 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_45.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %236 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %237 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %238 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %239 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_14.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %240 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %241 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %242 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_26.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %243 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %244 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %245 = phi_dt.tensor_map_get_tensor(%arg0) {name = "linear_0.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %246 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_51.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %247 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_41.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %248 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %249 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %250 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %251 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %252 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %253 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %254 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %255 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %256 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %257 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.w_1"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %258 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %259 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %260 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %261 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %262 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %263 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_17.w_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %264 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.w_2"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %265 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %266 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.b_0"} -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %267 = "phi_dt.create_context.gpu"() : () -> !phi.context<GPU>
    %268 = "phi_gpu.conv2d_infer.float32.any"(%267, %86, %86) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [3 : i32, 3 : i32], strides = [2 : i32, 2 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0, %out1, %out2 = "phi_gpu.batch_norm_infer.float32.any"(%267, %268, %259, %189, %115, %81) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %269 = "phi_gpu.relu.float32.any"(%267, %out0) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %270 = "phi_gpu.pool2d.float32.any"(%267, %269) {adaptive = false, ceil_mode = false, data_format = "NCHW", exclusive = true, global_pooling = false, ksize = [3 : i32, 3 : i32], padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], pooling_type = "max", strides = [2 : i32, 2 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %271 = "phi_gpu.conv2d_infer.float32.any"(%267, %270, %11) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_0, %out1_1, %out2_2 = "phi_gpu.batch_norm_infer.float32.any"(%267, %271, %150, %107, %144, %202) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %272 = "phi_gpu.relu.float32.any"(%267, %out0_0) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %273 = "phi_gpu.conv2d_infer.float32.any"(%267, %272, %29) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_3, %out1_4, %out2_5 = "phi_gpu.batch_norm_infer.float32.any"(%267, %273, %205, %159, %89, %114) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %274 = "phi_gpu.relu.float32.any"(%267, %out0_3) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %275 = "phi_gpu.conv2d_infer.float32.any"(%267, %274, %99) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_6, %out1_7, %out2_8 = "phi_gpu.batch_norm_infer.float32.any"(%267, %275, %168, %135, %257, %230) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %276 = "phi_gpu.conv2d_infer.float32.any"(%267, %270, %154) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_9, %out1_10, %out2_11 = "phi_gpu.batch_norm_infer.float32.any"(%267, %276, %79, %131, %69, %83) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %277 = "phi_gpu.add_raw.float32.any"(%267, %out0_6, %out0_9) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %278 = "phi_gpu.relu.float32.any"(%267, %277) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %279 = "phi_gpu.conv2d_infer.float32.any"(%267, %278, %231) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_12, %out1_13, %out2_14 = "phi_gpu.batch_norm_infer.float32.any"(%267, %279, %37, %261, %171, %4) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %280 = "phi_gpu.relu.float32.any"(%267, %out0_12) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %281 = "phi_gpu.conv2d_infer.float32.any"(%267, %280, %2) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_15, %out1_16, %out2_17 = "phi_gpu.batch_norm_infer.float32.any"(%267, %281, %206, %218, %236, %6) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %282 = "phi_gpu.relu.float32.any"(%267, %out0_15) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %283 = "phi_gpu.conv2d_infer.float32.any"(%267, %282, %44) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_18, %out1_19, %out2_20 = "phi_gpu.batch_norm_infer.float32.any"(%267, %283, %162, %50, %43, %213) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %284 = "phi_gpu.add_raw.float32.any"(%267, %out0_18, %278) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %285 = "phi_gpu.relu.float32.any"(%267, %284) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %286 = "phi_gpu.conv2d_infer.float32.any"(%267, %285, %34) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_21, %out1_22, %out2_23 = "phi_gpu.batch_norm_infer.float32.any"(%267, %286, %97, %14, %225, %138) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %287 = "phi_gpu.relu.float32.any"(%267, %out0_21) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %288 = "phi_gpu.conv2d_infer.float32.any"(%267, %287, %134) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_24, %out1_25, %out2_26 = "phi_gpu.batch_norm_infer.float32.any"(%267, %288, %222, %198, %42, %136) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %289 = "phi_gpu.relu.float32.any"(%267, %out0_24) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %290 = "phi_gpu.conv2d_infer.float32.any"(%267, %289, %219) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_27, %out1_28, %out2_29 = "phi_gpu.batch_norm_infer.float32.any"(%267, %290, %40, %194, %191, %82) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %291 = "phi_gpu.add_raw.float32.any"(%267, %out0_27, %285) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %292 = "phi_gpu.relu.float32.any"(%267, %291) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %293 = "phi_gpu.conv2d_infer.float32.any"(%267, %292, %197) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_30, %out1_31, %out2_32 = "phi_gpu.batch_norm_infer.float32.any"(%267, %293, %229, %85, %51, %186) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %294 = "phi_gpu.relu.float32.any"(%267, %out0_30) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %295 = "phi_gpu.conv2d_infer.float32.any"(%267, %294, %84) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_33, %out1_34, %out2_35 = "phi_gpu.batch_norm_infer.float32.any"(%267, %295, %3, %54, %88, %226) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %296 = "phi_gpu.relu.float32.any"(%267, %out0_33) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %297 = "phi_gpu.conv2d_infer.float32.any"(%267, %296, %239) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_36, %out1_37, %out2_38 = "phi_gpu.batch_norm_infer.float32.any"(%267, %297, %196, %121, %204, %195) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %298 = "phi_gpu.conv2d_infer.float32.any"(%267, %292, %74) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [2 : i32, 2 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_39, %out1_40, %out2_41 = "phi_gpu.batch_norm_infer.float32.any"(%267, %298, %254, %262, %8, %106) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %299 = "phi_gpu.add_raw.float32.any"(%267, %out0_36, %out0_39) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %300 = "phi_gpu.relu.float32.any"(%267, %299) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %301 = "phi_gpu.conv2d_infer.float32.any"(%267, %300, %175) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_42, %out1_43, %out2_44 = "phi_gpu.batch_norm_infer.float32.any"(%267, %301, %151, %100, %13, %109) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %302 = "phi_gpu.relu.float32.any"(%267, %out0_42) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %303 = "phi_gpu.conv2d_infer.float32.any"(%267, %302, %199) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_45, %out1_46, %out2_47 = "phi_gpu.batch_norm_infer.float32.any"(%267, %303, %72, %77, %210, %101) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %304 = "phi_gpu.relu.float32.any"(%267, %out0_45) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %305 = "phi_gpu.conv2d_infer.float32.any"(%267, %304, %263) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_48, %out1_49, %out2_50 = "phi_gpu.batch_norm_infer.float32.any"(%267, %305, %129, %149, %170, %112) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %306 = "phi_gpu.add_raw.float32.any"(%267, %out0_48, %300) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %307 = "phi_gpu.relu.float32.any"(%267, %306) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %308 = "phi_gpu.conv2d_infer.float32.any"(%267, %307, %25) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_51, %out1_52, %out2_53 = "phi_gpu.batch_norm_infer.float32.any"(%267, %308, %98, %176, %153, %216) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %309 = "phi_gpu.relu.float32.any"(%267, %out0_51) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %310 = "phi_gpu.conv2d_infer.float32.any"(%267, %309, %160) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_54, %out1_55, %out2_56 = "phi_gpu.batch_norm_infer.float32.any"(%267, %310, %145, %130, %192, %142) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %311 = "phi_gpu.relu.float32.any"(%267, %out0_54) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %312 = "phi_gpu.conv2d_infer.float32.any"(%267, %311, %108) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_57, %out1_58, %out2_59 = "phi_gpu.batch_norm_infer.float32.any"(%267, %312, %117, %207, %174, %71) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %313 = "phi_gpu.add_raw.float32.any"(%267, %out0_57, %307) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %314 = "phi_gpu.relu.float32.any"(%267, %313) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %315 = "phi_gpu.conv2d_infer.float32.any"(%267, %314, %155) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_60, %out1_61, %out2_62 = "phi_gpu.batch_norm_infer.float32.any"(%267, %315, %240, %12, %133, %128) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %316 = "phi_gpu.relu.float32.any"(%267, %out0_60) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %317 = "phi_gpu.conv2d_infer.float32.any"(%267, %316, %20) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_63, %out1_64, %out2_65 = "phi_gpu.batch_norm_infer.float32.any"(%267, %317, %184, %180, %68, %127) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %318 = "phi_gpu.relu.float32.any"(%267, %out0_63) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %319 = "phi_gpu.conv2d_infer.float32.any"(%267, %318, %111) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_66, %out1_67, %out2_68 = "phi_gpu.batch_norm_infer.float32.any"(%267, %319, %152, %243, %73, %146) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %320 = "phi_gpu.add_raw.float32.any"(%267, %out0_66, %314) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %321 = "phi_gpu.relu.float32.any"(%267, %320) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %322 = "phi_gpu.conv2d_infer.float32.any"(%267, %321, %113) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_69, %out1_70, %out2_71 = "phi_gpu.batch_norm_infer.float32.any"(%267, %322, %179, %93, %126, %165) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %323 = "phi_gpu.relu.float32.any"(%267, %out0_69) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %324 = "phi_gpu.conv2d_infer.float32.any"(%267, %323, %242) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_72, %out1_73, %out2_74 = "phi_gpu.batch_norm_infer.float32.any"(%267, %324, %17, %91, %185, %167) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %325 = "phi_gpu.relu.float32.any"(%267, %out0_72) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %326 = "phi_gpu.conv2d_infer.float32.any"(%267, %325, %21) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_75, %out1_76, %out2_77 = "phi_gpu.batch_norm_infer.float32.any"(%267, %326, %143, %63, %7, %173) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %327 = "phi_gpu.conv2d_infer.float32.any"(%267, %321, %105) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [2 : i32, 2 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_78, %out1_79, %out2_80 = "phi_gpu.batch_norm_infer.float32.any"(%267, %327, %182, %266, %251, %67) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %328 = "phi_gpu.add_raw.float32.any"(%267, %out0_75, %out0_78) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %329 = "phi_gpu.relu.float32.any"(%267, %328) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %330 = "phi_gpu.conv2d_infer.float32.any"(%267, %329, %157) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_81, %out1_82, %out2_83 = "phi_gpu.batch_norm_infer.float32.any"(%267, %330, %241, %22, %61, %172) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %331 = "phi_gpu.relu.float32.any"(%267, %out0_81) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %332 = "phi_gpu.conv2d_infer.float32.any"(%267, %331, %15) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_84, %out1_85, %out2_86 = "phi_gpu.batch_norm_infer.float32.any"(%267, %332, %60, %200, %57, %38) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %333 = "phi_gpu.relu.float32.any"(%267, %out0_84) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %334 = "phi_gpu.conv2d_infer.float32.any"(%267, %333, %75) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_87, %out1_88, %out2_89 = "phi_gpu.batch_norm_infer.float32.any"(%267, %334, %65, %87, %53, %95) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %335 = "phi_gpu.add_raw.float32.any"(%267, %out0_87, %329) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %336 = "phi_gpu.relu.float32.any"(%267, %335) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %337 = "phi_gpu.conv2d_infer.float32.any"(%267, %336, %158) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_90, %out1_91, %out2_92 = "phi_gpu.batch_norm_infer.float32.any"(%267, %337, %80, %19, %49, %252) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %338 = "phi_gpu.relu.float32.any"(%267, %out0_90) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %339 = "phi_gpu.conv2d_infer.float32.any"(%267, %338, %214) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_93, %out1_94, %out2_95 = "phi_gpu.batch_norm_infer.float32.any"(%267, %339, %70, %166, %102, %147) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %340 = "phi_gpu.relu.float32.any"(%267, %out0_93) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %341 = "phi_gpu.conv2d_infer.float32.any"(%267, %340, %233) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_96, %out1_97, %out2_98 = "phi_gpu.batch_norm_infer.float32.any"(%267, %341, %66, %94, %39, %163) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %342 = "phi_gpu.add_raw.float32.any"(%267, %out0_96, %336) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %343 = "phi_gpu.relu.float32.any"(%267, %342) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %344 = "phi_gpu.conv2d_infer.float32.any"(%267, %343, %124) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_99, %out1_100, %out2_101 = "phi_gpu.batch_norm_infer.float32.any"(%267, %344, %256, %265, %31, %209) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %345 = "phi_gpu.relu.float32.any"(%267, %out0_99) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %346 = "phi_gpu.conv2d_infer.float32.any"(%267, %345, %16) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_102, %out1_103, %out2_104 = "phi_gpu.batch_norm_infer.float32.any"(%267, %346, %139, %248, %96, %181) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %347 = "phi_gpu.relu.float32.any"(%267, %out0_102) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %348 = "phi_gpu.conv2d_infer.float32.any"(%267, %347, %211) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_105, %out1_106, %out2_107 = "phi_gpu.batch_norm_infer.float32.any"(%267, %348, %260, %78, %56, %58) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %349 = "phi_gpu.add_raw.float32.any"(%267, %out0_105, %343) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %350 = "phi_gpu.relu.float32.any"(%267, %349) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %351 = "phi_gpu.conv2d_infer.float32.any"(%267, %350, %24) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_108, %out1_109, %out2_110 = "phi_gpu.batch_norm_infer.float32.any"(%267, %351, %76, %156, %238, %0) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %352 = "phi_gpu.relu.float32.any"(%267, %out0_108) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %353 = "phi_gpu.conv2d_infer.float32.any"(%267, %352, %26) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_111, %out1_112, %out2_113 = "phi_gpu.batch_norm_infer.float32.any"(%267, %353, %10, %161, %116, %188) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %354 = "phi_gpu.relu.float32.any"(%267, %out0_111) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %355 = "phi_gpu.conv2d_infer.float32.any"(%267, %354, %27) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_114, %out1_115, %out2_116 = "phi_gpu.batch_norm_infer.float32.any"(%267, %355, %52, %208, %140, %264) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %356 = "phi_gpu.add_raw.float32.any"(%267, %out0_114, %350) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %357 = "phi_gpu.relu.float32.any"(%267, %356) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %358 = "phi_gpu.conv2d_infer.float32.any"(%267, %357, %46) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_117, %out1_118, %out2_119 = "phi_gpu.batch_norm_infer.float32.any"(%267, %358, %258, %9, %5, %169) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %359 = "phi_gpu.relu.float32.any"(%267, %out0_117) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %360 = "phi_gpu.conv2d_infer.float32.any"(%267, %359, %247) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_120, %out1_121, %out2_122 = "phi_gpu.batch_norm_infer.float32.any"(%267, %360, %177, %255, %253, %193) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %361 = "phi_gpu.relu.float32.any"(%267, %out0_120) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %362 = "phi_gpu.conv2d_infer.float32.any"(%267, %361, %178) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_123, %out1_124, %out2_125 = "phi_gpu.batch_norm_infer.float32.any"(%267, %362, %47, %201, %48, %249) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %363 = "phi_gpu.add_raw.float32.any"(%267, %out0_123, %357) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %364 = "phi_gpu.relu.float32.any"(%267, %363) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %365 = "phi_gpu.conv2d_infer.float32.any"(%267, %364, %104) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_126, %out1_127, %out2_128 = "phi_gpu.batch_norm_infer.float32.any"(%267, %365, %164, %217, %110, %141) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %366 = "phi_gpu.relu.float32.any"(%267, %out0_126) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %367 = "phi_gpu.conv2d_infer.float32.any"(%267, %366, %235) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_129, %out1_130, %out2_131 = "phi_gpu.batch_norm_infer.float32.any"(%267, %367, %137, %125, %224, %35) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %368 = "phi_gpu.relu.float32.any"(%267, %out0_129) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %369 = "phi_gpu.conv2d_infer.float32.any"(%267, %368, %234) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_132, %out1_133, %out2_134 = "phi_gpu.batch_norm_infer.float32.any"(%267, %369, %227, %55, %123, %23) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %370 = "phi_gpu.conv2d_infer.float32.any"(%267, %364, %28) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [2 : i32, 2 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_135, %out1_136, %out2_137 = "phi_gpu.batch_norm_infer.float32.any"(%267, %370, %187, %36, %41, %250) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %371 = "phi_gpu.add_raw.float32.any"(%267, %out0_132, %out0_135) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %372 = "phi_gpu.relu.float32.any"(%267, %371) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %373 = "phi_gpu.conv2d_infer.float32.any"(%267, %372, %64) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_138, %out1_139, %out2_140 = "phi_gpu.batch_norm_infer.float32.any"(%267, %373, %1, %122, %118, %220) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %374 = "phi_gpu.relu.float32.any"(%267, %out0_138) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %375 = "phi_gpu.conv2d_infer.float32.any"(%267, %374, %203) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_141, %out1_142, %out2_143 = "phi_gpu.batch_norm_infer.float32.any"(%267, %375, %237, %120, %212, %92) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %376 = "phi_gpu.relu.float32.any"(%267, %out0_141) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %377 = "phi_gpu.conv2d_infer.float32.any"(%267, %376, %32) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_144, %out1_145, %out2_146 = "phi_gpu.batch_norm_infer.float32.any"(%267, %377, %244, %59, %183, %228) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %378 = "phi_gpu.add_raw.float32.any"(%267, %out0_144, %372) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %379 = "phi_gpu.relu.float32.any"(%267, %378) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %380 = "phi_gpu.conv2d_infer.float32.any"(%267, %379, %190) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_147, %out1_148, %out2_149 = "phi_gpu.batch_norm_infer.float32.any"(%267, %380, %119, %103, %18, %45) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %381 = "phi_gpu.relu.float32.any"(%267, %out0_147) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %382 = "phi_gpu.conv2d_infer.float32.any"(%267, %381, %246) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_150, %out1_151, %out2_152 = "phi_gpu.batch_norm_infer.float32.any"(%267, %382, %148, %232, %221, %62) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %383 = "phi_gpu.relu.float32.any"(%267, %out0_150) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %384 = "phi_gpu.conv2d_infer.float32.any"(%267, %383, %132) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %out0_153, %out1_154, %out2_155 = "phi_gpu.batch_norm_infer.float32.any"(%267, %384, %215, %90, %33, %223) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>)
    %385 = "phi_gpu.add_raw.float32.any"(%267, %out0_153, %379) {axis = -1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %386 = "phi_gpu.relu.float32.any"(%267, %385) : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %387 = "phi_gpu.pool2d.float32.any"(%267, %386) {adaptive = true, ceil_mode = false, data_format = "NCHW", exclusive = true, global_pooling = false, ksize = [1 : i32, 1 : i32], padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], pooling_type = "avg", strides = [1 : i32, 1 : i32]} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %388 = "phi_gpu.flatten.float32.any"(%267, %387) {start_axis = 1 : si32, stop_axis = 3 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %389 = "phi_gpu.matmul.float32.any"(%267, %388, %245) {trans_x = false, trans_y = false} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %390 = "phi_gpu.add_raw.float32.any"(%267, %389, %30) {axis = 1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    infrt.return %390 : !infrt.dense_tensor<GPU, FP32, NCHW>
  }

  func @main() {
    %ctx = "phi_dt.create_context.gpu"() : () -> !phi.context<GPU>
    %in = "phi_dt.create_inited_dense_tensor.gpu.f32" (%ctx) {value = 12.0 : f32, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[1, 3, 256, 256]}: (!phi.context<GPU>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>)


    %map = phi_dt.load_combined_params_to_gpu(){model_path="./resnet50/model.pdmodel",params_path="./resnet50/model.pdiparams"}
    %2 = infrt.call@main_graph(%map, %in) : (!phi.dense_tensor_map, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>

    %5 = "phi_dt.memcpy.gpu"(%2, %ctx) {d2h = true}:(!infrt.dense_tensor<GPU, FP32, NCHW>, !phi.context<GPU>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    phi_dt.print_tensor (%5 : !infrt.dense_tensor<CPU, FP32, NCHW>)

    infrt.return
  }
}
