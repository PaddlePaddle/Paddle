#!/bin/bash

function base_image(){
  if [[ ${ref_CUDA_MAJOR} == "11.2" ]];then
    dockerfile_name="Dockerfile-112"
    sed "s#<baseimg>#nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.2/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
    sed -i "s#gcc121#gcc82#g" ${dockerfile_name}
    sed -i "s#gcc-12.1#gcc-8.2#g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8034#g' ${dockerfile_name}
    sed -i 's#cudnn841#cudnn821#g' ${dockerfile_name}
    sed -i 's#CUDNN_VERSION=8.4.1#CUDNN_VERSION=8.2.1#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "11.6" ]];then
    dockerfile_name="Dockerfile-116"
    sed "s#<baseimg>#nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
    sed -i "s#gcc121#gcc82#g" ${dockerfile_name}
    sed -i "s#gcc-12.1#gcc-8.2#g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8431#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "11.7" ]];then
    dockerfile_name="Dockerfile-117"
    sed "s#<baseimg>#nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
    sed -i "s#gcc121#gcc82#g" ${dockerfile_name}
    sed -i "s#gcc-12.1#gcc-8.2#g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8431#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "11.8" ]];then
    dockerfile_name="Dockerfile-118"
    sed "s#<baseimg>#nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
    sed -i "s#gcc121#gcc82#g" ${dockerfile_name}
    sed -i "s#gcc-12.1#gcc-8.2#g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8531#g' ${dockerfile_name}
    sed -i 's#cudnn841#cudnn860#g' ${dockerfile_name}
    sed -i 's#CUDNN_VERSION=8.4.1#CUDNN_VERSION=8.6.0#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "12.0" ]];then
    dockerfile_name="Dockerfile-120"
    sed "s#<baseimg>#nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.0/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-12.0/bin:\$PATH #g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8616#g' ${dockerfile_name}
    sed -i 's#cudnn841#cudnn891#g' ${dockerfile_name}
    sed -i 's#CUDNN_VERSION=8.4.1#CUDNN_VERSION=8.9.1#g' ${dockerfile_name}
  else
    echo "Dockerfile ERROR!!!"
    exit 1
  fi

}

base_image
