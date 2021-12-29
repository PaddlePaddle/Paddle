// CHECK-LABEL: @main
func @main() -> tensor<?xf32> {
  %a = "pd.feed"() : () -> tensor<?x3x256x256xf32>
  %filter = "pd.constant"(){value = dense<1.000000e+00> : tensor<3x64x3x3xf32>} : () -> tensor<3x64x3x3xf32> 
  %bias = "pd.constant"(){value = dense<1.000000e+00> : tensor<64xf32>} : () -> tensor<64xf32>

  %scale = "pd.constant"(){value = dense<1.000000e+00> : tensor<64xf32>} : () -> tensor<64xf32>
  %bias2 = "pd.constant"(){value = dense<1.000000e+00> : tensor<64xf32>} : () -> tensor<64xf32>
  %mean = "pd.constant"(){value = dense<1.000000e+00> : tensor<64xf32>} : () -> tensor<64xf32>
  %var = "pd.constant"(){value = dense<1.000000e+00> : tensor<64xf32>} : () -> tensor<64xf32>

  %c = "pd.conv2d"(%a, %filter, %bias) {} : (tensor<?x3x256x256xf32>, tensor<3x64x3x3xf32>, tensor<64xf32>) -> tensor<?x3x256x256xf32>
  %d = "pd.batch_norm"(%c, %scale, %bias2, %mean, %var) {} : (tensor<?x3x256x256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x3x256x256xf32>
  infrt.return %d : tensor<?x3x256x256xf32>
}