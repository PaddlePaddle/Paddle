# This generate kernel template script is referenced from 
# https://github.com/facebookresearch/xformers/blob/main/xformers/csrc/attention/cuda/fmha/kernels/generate_kernels.sh

#!/bin/bash
set -ex
rm -f forward_*.cu
IFS=","

# FORWARD
kernel="FORWARD"
kernel_lower=`echo "\$kernel" | awk '{print tolower($0)}'`

for mask_broadcast in "false" "true"; do
    [[ $mask_broadcast = "true" ]] && mask_broadcast_suffix="_maskbroadcast" || mask_broadcast_suffix=""
    for add_mask in "false" "true"; do
        [[ $add_mask = "true" ]] && add_mask_suffix="_addmask" || add_mask_suffix=""
        for aligned in "false" "true"; do
            [[ $aligned = "true" ]] && aligned_suffix="_aligned" || aligned_suffix=""
            for dtype_name in "f32" "f16" ; do
                case "$dtype_name" in
                    "f32") dtype="float" ;;
                    "f16") dtype="cutlass::half_t" ;;
                esac
                FNAME="${kernel_lower}_${dtype_name}${aligned_suffix}${add_mask_suffix}${mask_broadcast_suffix}.cu"
                echo $FNAME
                cat <<EOF > $FNAME
// This file is auto-generated. See "generate_kernels.sh"
#include "forward.h"
EOF
                for sm in 70 75 80; do
                    echo "INSTANTIATE_ATTENTION_KERNEL_${kernel}_SM${sm}($dtype, $aligned, 32, 128, true, $add_mask, $mask_broadcast);" >> $FNAME
                    echo "INSTANTIATE_ATTENTION_KERNEL_${kernel}_SM${sm}($dtype, $aligned, 32, 128, false, $add_mask, $mask_broadcast);" >> $FNAME
                    echo "INSTANTIATE_ATTENTION_KERNEL_${kernel}_SM${sm}($dtype, $aligned, 64, 64, true, $add_mask, $mask_broadcast);" >> $FNAME
                done;
            done;
        done;
    done;
done
