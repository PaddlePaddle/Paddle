#!/bin/bash
set -e
cd `dirname $0`
on_off=('ON' 'OFF')
images=('nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04' 'nvidia/cuda:7.5-cudnn5-devel-centos7' 'ubuntu:14.04' 'centos:7')
build_script=('build.sh' 'build_centos.sh')
cpu_tag=('gpu' 'cpu')
image_tag=('-ubuntu14.04' '-centos7')
avx_tag=('' '-noavx')

# gpu/cpu
for i in 0 1; do
# ubuntu/centos
    for j in 0 1; do
# avx/noavx
        for k in 0 1; do
            m4 -DBUILD_SCRIPT=${build_script[j]} -DPADDLE_WITH_GPU=${on_off[i]} -DPADDLE_IS_DEVEL=OFF \
            -DPADDLE_WITH_DEMO=OFF -DPADDLE_BASE_IMAGE=${images[2*i+j]} -DPADDLE_WITH_AVX=${on_off[k]}\
            Dockerfile.m4 > Dockerfile.${cpu_tag[i]}${avx_tag[k]}${image_tag[j]}

            m4 -DBUILD_SCRIPT=${build_script[j]} -DPADDLE_WITH_GPU=${on_off[i]} -DPADDLE_IS_DEVEL=ON \
	    -DPADDLE_WITH_DEMO=OFF -DPADDLE_BASE_IMAGE=${images[2*i+j]} -DPADDLE_WITH_AVX=${on_off[k]}\
            Dockerfile.m4 > Dockerfile.${cpu_tag[i]}${avx_tag[k]}-devel${image_tag[j]}

            m4 -DBUILD_SCRIPT=${build_script[j]} -DPADDLE_WITH_GPU=${on_off[i]} -DPADDLE_IS_DEVEL=ON \
	    -DPADDLE_WITH_DEMO=ON -DPADDLE_BASE_IMAGE=${images[2*i+j]} -DPADDLE_WITH_AVX=${on_off[k]}\
            Dockerfile.m4 > Dockerfile.${cpu_tag[i]}${avx_tag[k]}-demo${image_tag[j]}
        done
    done
done


