module  {
  func @main_graph(%arg0: !phi.dense_tensor_map, %arg1: !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY> {
    %0 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %1 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %2 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_6.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %3 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %4 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %5 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %6 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %7 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %8 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %9 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %10 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %11 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_2.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %12 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %13 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %14 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %15 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_29.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %16 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_35.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %17 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %18 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %19 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %20 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_22.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %21 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_27.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %22 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %23 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %24 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_37.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %25 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_18.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %26 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_38.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %27 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_39.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %28 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_43.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %29 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_3.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %30 = phi_dt.tensor_map_get_tensor(%arg0) {name = "linear_0.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %31 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %32 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_49.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %33 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %34 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_8.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %35 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %36 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %37 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %38 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %39 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %40 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %41 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %42 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %43 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %44 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_7.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %45 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %46 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_40.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %47 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %48 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %49 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %50 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %51 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %52 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %53 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %54 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %55 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %56 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %57 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %58 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %59 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %60 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %61 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %62 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %63 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %64 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_47.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %65 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %66 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %67 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %68 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %69 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %70 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %71 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %72 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %73 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %74 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_11.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %75 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_30.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %76 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %77 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %78 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %79 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %80 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %81 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %82 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %83 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %84 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_13.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %85 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %86 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_0.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %87 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %88 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %89 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %90 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %91 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %92 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %93 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %94 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %95 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_30.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %96 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %97 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %98 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %99 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_4.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %100 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %101 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %102 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %103 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %104 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_44.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %105 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_24.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %106 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %107 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %108 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_20.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %109 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %110 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %111 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_23.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %112 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %113 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_25.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %114 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %115 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %116 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %117 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %118 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %119 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_50.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %120 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %121 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %122 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %123 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %124 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_34.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %125 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %126 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %127 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %128 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %129 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %130 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %131 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_1.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %132 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_52.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %133 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %134 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_9.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %135 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %136 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %137 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %138 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %139 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %140 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %141 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %142 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %143 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %144 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %145 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %146 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %147 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %148 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %149 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %150 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %151 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_15.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %152 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %153 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %154 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_1.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %155 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_21.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %156 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %157 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_28.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %158 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_31.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %159 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %160 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_19.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %161 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %162 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %163 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_33.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %164 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %165 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %166 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_32.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %167 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %168 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %169 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %170 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_17.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %171 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %172 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %173 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_27.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %174 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %175 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_15.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %176 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %177 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %178 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_42.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %179 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_25.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %180 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %181 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %182 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %183 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %184 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_22.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %185 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_26.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %186 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %187 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %188 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_38.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %189 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %190 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_50.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %191 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %192 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_19.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %193 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %194 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_10.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %195 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %196 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %197 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_12.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %198 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %199 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_16.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %200 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_29.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %201 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %202 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_2.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %203 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_48.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %204 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_14.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %205 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_3.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %206 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %207 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_20.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %208 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %209 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %210 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_16.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %211 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_36.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %212 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %213 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_7.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %214 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_32.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %215 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %216 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_18.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %217 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_44.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %218 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %219 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_10.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %220 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_47.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %221 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %222 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_9.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %223 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_52.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %224 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_45.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %225 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_8.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %226 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_13.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %227 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_46.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %228 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %229 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_12.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %230 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %231 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_5.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %232 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_51.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %233 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_33.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %234 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_46.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %235 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_45.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %236 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_6.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %237 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_48.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %238 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_37.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %239 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_14.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %240 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_21.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %241 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_28.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %242 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_26.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %243 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_23.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %244 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_49.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %245 = phi_dt.tensor_map_get_tensor(%arg0) {name = "linear_0.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %246 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_51.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %247 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_41.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %248 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_35.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %249 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_42.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %250 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_43.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %251 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %252 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_31.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %253 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %254 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %255 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_41.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %256 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %257 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_4.w_1"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %258 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_40.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %259 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_0.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %260 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_36.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %261 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_5.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %262 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_11.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %263 = phi_dt.tensor_map_get_tensor(%arg0) {name = "conv2d_17.w_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %264 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_39.w_2"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %265 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_34.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %266 = phi_dt.tensor_map_get_tensor(%arg0) {name = "batch_norm2d_24.b_0"} -> !infrt.dense_tensor<CPU, FP32, ANY>
    %267 = "pd.conv2d"(%86, %arg1) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [3 : i32, 3 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y, %MeanOut, %VarianceOut = "pd.batch_norm"(%189, %115, %259, %81, %267) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %268 = "pd.relu"(%Y) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %269 = "pd.pool2d"(%268) {adaptive = false, ceil_mode = false, data_format = "NCHW", exclusive = true, global_pooling = false, ksize = [3 : i32, 3 : i32], padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %270 = "pd.conv2d"(%11, %269) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_0, %MeanOut_1, %VarianceOut_2 = "pd.batch_norm"(%107, %144, %150, %202, %270) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %271 = "pd.relu"(%Y_0) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %272 = "pd.conv2d"(%29, %271) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_3, %MeanOut_4, %VarianceOut_5 = "pd.batch_norm"(%159, %89, %205, %114, %272) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %273 = "pd.relu"(%Y_3) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %274 = "pd.conv2d"(%99, %273) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_6, %MeanOut_7, %VarianceOut_8 = "pd.batch_norm"(%135, %257, %168, %230, %274) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %275 = "pd.conv2d"(%154, %269) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_9, %MeanOut_10, %VarianceOut_11 = "pd.batch_norm"(%131, %69, %79, %83, %275) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %276 = "pd.elementwise_add"(%Y_6, %Y_9) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %277 = "pd.relu"(%276) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %278 = "pd.conv2d"(%231, %277) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_12, %MeanOut_13, %VarianceOut_14 = "pd.batch_norm"(%261, %171, %37, %4, %278) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %279 = "pd.relu"(%Y_12) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %280 = "pd.conv2d"(%2, %279) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_15, %MeanOut_16, %VarianceOut_17 = "pd.batch_norm"(%218, %236, %206, %6, %280) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %281 = "pd.relu"(%Y_15) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %282 = "pd.conv2d"(%44, %281) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_18, %MeanOut_19, %VarianceOut_20 = "pd.batch_norm"(%50, %43, %162, %213, %282) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %283 = "pd.elementwise_add"(%Y_18, %277) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %284 = "pd.relu"(%283) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %285 = "pd.conv2d"(%34, %284) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_21, %MeanOut_22, %VarianceOut_23 = "pd.batch_norm"(%14, %225, %97, %138, %285) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %286 = "pd.relu"(%Y_21) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %287 = "pd.conv2d"(%134, %286) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_24, %MeanOut_25, %VarianceOut_26 = "pd.batch_norm"(%198, %42, %222, %136, %287) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %288 = "pd.relu"(%Y_24) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %289 = "pd.conv2d"(%219, %288) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_27, %MeanOut_28, %VarianceOut_29 = "pd.batch_norm"(%194, %191, %40, %82, %289) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %290 = "pd.elementwise_add"(%Y_27, %284) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %291 = "pd.relu"(%290) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %292 = "pd.conv2d"(%197, %291) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_30, %MeanOut_31, %VarianceOut_32 = "pd.batch_norm"(%85, %51, %229, %186, %292) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %293 = "pd.relu"(%Y_30) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %294 = "pd.conv2d"(%84, %293) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_33, %MeanOut_34, %VarianceOut_35 = "pd.batch_norm"(%54, %88, %3, %226, %294) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %295 = "pd.relu"(%Y_33) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %296 = "pd.conv2d"(%239, %295) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_36, %MeanOut_37, %VarianceOut_38 = "pd.batch_norm"(%121, %204, %196, %195, %296) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %297 = "pd.conv2d"(%74, %291) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_39, %MeanOut_40, %VarianceOut_41 = "pd.batch_norm"(%262, %8, %254, %106, %297) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %298 = "pd.elementwise_add"(%Y_36, %Y_39) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %299 = "pd.relu"(%298) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %300 = "pd.conv2d"(%175, %299) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_42, %MeanOut_43, %VarianceOut_44 = "pd.batch_norm"(%100, %13, %151, %109, %300) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %301 = "pd.relu"(%Y_42) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %302 = "pd.conv2d"(%199, %301) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_45, %MeanOut_46, %VarianceOut_47 = "pd.batch_norm"(%77, %210, %72, %101, %302) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %303 = "pd.relu"(%Y_45) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %304 = "pd.conv2d"(%263, %303) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_48, %MeanOut_49, %VarianceOut_50 = "pd.batch_norm"(%149, %170, %129, %112, %304) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %305 = "pd.elementwise_add"(%Y_48, %299) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %306 = "pd.relu"(%305) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %307 = "pd.conv2d"(%25, %306) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_51, %MeanOut_52, %VarianceOut_53 = "pd.batch_norm"(%176, %153, %98, %216, %307) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %308 = "pd.relu"(%Y_51) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %309 = "pd.conv2d"(%160, %308) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_54, %MeanOut_55, %VarianceOut_56 = "pd.batch_norm"(%130, %192, %145, %142, %309) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %310 = "pd.relu"(%Y_54) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %311 = "pd.conv2d"(%108, %310) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_57, %MeanOut_58, %VarianceOut_59 = "pd.batch_norm"(%207, %174, %117, %71, %311) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %312 = "pd.elementwise_add"(%Y_57, %306) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %313 = "pd.relu"(%312) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %314 = "pd.conv2d"(%155, %313) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_60, %MeanOut_61, %VarianceOut_62 = "pd.batch_norm"(%12, %133, %240, %128, %314) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %315 = "pd.relu"(%Y_60) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %316 = "pd.conv2d"(%20, %315) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_63, %MeanOut_64, %VarianceOut_65 = "pd.batch_norm"(%180, %68, %184, %127, %316) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %317 = "pd.relu"(%Y_63) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %318 = "pd.conv2d"(%111, %317) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_66, %MeanOut_67, %VarianceOut_68 = "pd.batch_norm"(%243, %73, %152, %146, %318) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %319 = "pd.elementwise_add"(%Y_66, %313) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %320 = "pd.relu"(%319) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %321 = "pd.conv2d"(%113, %320) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_69, %MeanOut_70, %VarianceOut_71 = "pd.batch_norm"(%93, %126, %179, %165, %321) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %322 = "pd.relu"(%Y_69) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %323 = "pd.conv2d"(%242, %322) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_72, %MeanOut_73, %VarianceOut_74 = "pd.batch_norm"(%91, %185, %17, %167, %323) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %324 = "pd.relu"(%Y_72) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %325 = "pd.conv2d"(%21, %324) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_75, %MeanOut_76, %VarianceOut_77 = "pd.batch_norm"(%63, %7, %143, %173, %325) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %326 = "pd.conv2d"(%105, %320) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_78, %MeanOut_79, %VarianceOut_80 = "pd.batch_norm"(%266, %251, %182, %67, %326) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %327 = "pd.elementwise_add"(%Y_75, %Y_78) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %328 = "pd.relu"(%327) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %329 = "pd.conv2d"(%157, %328) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_81, %MeanOut_82, %VarianceOut_83 = "pd.batch_norm"(%22, %61, %241, %172, %329) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %330 = "pd.relu"(%Y_81) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %331 = "pd.conv2d"(%15, %330) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_84, %MeanOut_85, %VarianceOut_86 = "pd.batch_norm"(%200, %57, %60, %38, %331) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %332 = "pd.relu"(%Y_84) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %333 = "pd.conv2d"(%75, %332) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_87, %MeanOut_88, %VarianceOut_89 = "pd.batch_norm"(%87, %53, %65, %95, %333) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %334 = "pd.elementwise_add"(%Y_87, %328) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %335 = "pd.relu"(%334) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %336 = "pd.conv2d"(%158, %335) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_90, %MeanOut_91, %VarianceOut_92 = "pd.batch_norm"(%19, %49, %80, %252, %336) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %337 = "pd.relu"(%Y_90) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %338 = "pd.conv2d"(%214, %337) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_93, %MeanOut_94, %VarianceOut_95 = "pd.batch_norm"(%166, %102, %70, %147, %338) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %339 = "pd.relu"(%Y_93) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %340 = "pd.conv2d"(%233, %339) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_96, %MeanOut_97, %VarianceOut_98 = "pd.batch_norm"(%94, %39, %66, %163, %340) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %341 = "pd.elementwise_add"(%Y_96, %335) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %342 = "pd.relu"(%341) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %343 = "pd.conv2d"(%124, %342) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_99, %MeanOut_100, %VarianceOut_101 = "pd.batch_norm"(%265, %31, %256, %209, %343) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %344 = "pd.relu"(%Y_99) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %345 = "pd.conv2d"(%16, %344) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_102, %MeanOut_103, %VarianceOut_104 = "pd.batch_norm"(%248, %96, %139, %181, %345) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %346 = "pd.relu"(%Y_102) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %347 = "pd.conv2d"(%211, %346) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_105, %MeanOut_106, %VarianceOut_107 = "pd.batch_norm"(%78, %56, %260, %58, %347) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %348 = "pd.elementwise_add"(%Y_105, %342) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %349 = "pd.relu"(%348) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %350 = "pd.conv2d"(%24, %349) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_108, %MeanOut_109, %VarianceOut_110 = "pd.batch_norm"(%156, %238, %76, %0, %350) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %351 = "pd.relu"(%Y_108) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %352 = "pd.conv2d"(%26, %351) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_111, %MeanOut_112, %VarianceOut_113 = "pd.batch_norm"(%161, %116, %10, %188, %352) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %353 = "pd.relu"(%Y_111) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %354 = "pd.conv2d"(%27, %353) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_114, %MeanOut_115, %VarianceOut_116 = "pd.batch_norm"(%208, %140, %52, %264, %354) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %355 = "pd.elementwise_add"(%Y_114, %349) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %356 = "pd.relu"(%355) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %357 = "pd.conv2d"(%46, %356) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_117, %MeanOut_118, %VarianceOut_119 = "pd.batch_norm"(%9, %5, %258, %169, %357) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %358 = "pd.relu"(%Y_117) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %359 = "pd.conv2d"(%247, %358) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_120, %MeanOut_121, %VarianceOut_122 = "pd.batch_norm"(%255, %253, %177, %193, %359) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %360 = "pd.relu"(%Y_120) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %361 = "pd.conv2d"(%178, %360) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_123, %MeanOut_124, %VarianceOut_125 = "pd.batch_norm"(%201, %48, %47, %249, %361) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %362 = "pd.elementwise_add"(%Y_123, %356) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %363 = "pd.relu"(%362) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %364 = "pd.conv2d"(%104, %363) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_126, %MeanOut_127, %VarianceOut_128 = "pd.batch_norm"(%217, %110, %164, %141, %364) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %365 = "pd.relu"(%Y_126) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %366 = "pd.conv2d"(%235, %365) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_129, %MeanOut_130, %VarianceOut_131 = "pd.batch_norm"(%125, %224, %137, %35, %366) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %367 = "pd.relu"(%Y_129) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %368 = "pd.conv2d"(%234, %367) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_132, %MeanOut_133, %VarianceOut_134 = "pd.batch_norm"(%55, %123, %227, %23, %368) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %369 = "pd.conv2d"(%28, %363) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_135, %MeanOut_136, %VarianceOut_137 = "pd.batch_norm"(%36, %41, %187, %250, %369) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %370 = "pd.elementwise_add"(%Y_132, %Y_135) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %371 = "pd.relu"(%370) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %372 = "pd.conv2d"(%64, %371) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_138, %MeanOut_139, %VarianceOut_140 = "pd.batch_norm"(%122, %118, %1, %220, %372) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %373 = "pd.relu"(%Y_138) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %374 = "pd.conv2d"(%203, %373) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_141, %MeanOut_142, %VarianceOut_143 = "pd.batch_norm"(%120, %212, %237, %92, %374) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %375 = "pd.relu"(%Y_141) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %376 = "pd.conv2d"(%32, %375) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_144, %MeanOut_145, %VarianceOut_146 = "pd.batch_norm"(%59, %183, %244, %228, %376) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %377 = "pd.elementwise_add"(%Y_144, %371) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %378 = "pd.relu"(%377) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %379 = "pd.conv2d"(%190, %378) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_147, %MeanOut_148, %VarianceOut_149 = "pd.batch_norm"(%103, %18, %119, %45, %379) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %380 = "pd.relu"(%Y_147) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %381 = "pd.conv2d"(%246, %380) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_150, %MeanOut_151, %VarianceOut_152 = "pd.batch_norm"(%232, %221, %148, %62, %381) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %382 = "pd.relu"(%Y_150) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %383 = "pd.conv2d"(%132, %382) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %Y_153, %MeanOut_154, %VarianceOut_155 = "pd.batch_norm"(%90, %33, %215, %223, %383) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>)
    %384 = "pd.elementwise_add"(%Y_153, %378) {axis = -1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %385 = "pd.relu"(%384) : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %386 = "pd.pool2d"(%385) {adaptive = true, ceil_mode = false, data_format = "NCHW", exclusive = true, global_pooling = false, ksize = [1 : i32, 1 : i32], padding_algorithm = "EXPLICIT", paddings = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %387 = "pd.flatten_contiguous_range"(%386) {start_axis = 1 : si32, stop_axis = 3 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %388 = "pd.matmul_v2"(%387, %245) {trans_x = false, trans_y = false} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    %389 = "pd.elementwise_add"(%388, %30) {axis = 1 : si32} : (!infrt.dense_tensor<CPU, FP32, ANY>, !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    infrt.return %389 : !infrt.dense_tensor<CPU, FP32, ANY>
  }
}
