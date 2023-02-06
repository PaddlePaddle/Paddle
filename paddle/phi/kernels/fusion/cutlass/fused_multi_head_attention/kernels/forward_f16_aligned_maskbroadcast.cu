// This file is auto-generated. See "generate_kernels.sh"
#include "forward.h"
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70(cutlass::half_t, true, 32, 128, true, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70(cutlass::half_t, true, 32, 128, false, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70(cutlass::half_t, true, 64, 64, true, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70(cutlass::half_t, true, 64, 64, false, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75(cutlass::half_t, true, 32, 128, true, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75(cutlass::half_t, true, 32, 128, false, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75(cutlass::half_t, true, 64, 64, true, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75(cutlass::half_t, true, 64, 64, false, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80(cutlass::half_t, true, 32, 128, true, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80(cutlass::half_t, true, 32, 128, false, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80(cutlass::half_t, true, 64, 64, true, false, true);
INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80(cutlass::half_t, true, 64, 64, false, false, true);
