FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04

# you can get mirror list here:
# https://launchpad.net/ubuntu/+archivemirrors
ARG UBUNTU_MIRROR
RUN /bin/bash -c 'if [[ -n ${UBUNTU_MIRROR} ]]; then sed -i 's#http://archive.ubuntu.com/ubuntu#${UBUNTU_MIRROR}#g' /etc/apt/sources.list; fi'

RUN apt-get update && apt-get install -y python python-dev python-pip iputils-ping libgtk2.0-dev
RUN pip install -U kubernetes opencv-python

RUN pip install paddlepaddle
# if network is slowly, you may need to add proxy here.
# ENV https_proxy=
RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.cifar.train10()" | python'
RUN pip uninstall -y paddlepaddle
# unset proxy if it is setted.
# ENV https_proxy=""

# NOTE: By default CI built wheel packages turn WITH_DISTRIBUTE=OFF,
#       so we must build one with distribute support to install in this image.
ADD *.whl /
RUN pip install /*.whl && rm -f /*.whl
ENV LD_LIBRARY_PATH=/usr/local/lib

# tf k8s
RUN pip install tensorflow==1.4.0
ADD tf_k8s /usr/bin
RUN chmod +x /usr/bin/tf_k8s
ADD vgg16_tf.py /workspace/

# below lines may change a lot for debugging
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/paddle_k8s /usr/bin
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/k8s_tools.py /root
RUN chmod +x /usr/bin/paddle_k8s
ADD vgg16_fluid.py vgg16_v2.py /workspace/
