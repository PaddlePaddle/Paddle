#FROM python:2.7.14
FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04
RUN apt-get update && apt-get install -y python
RUN pip install -U kubernetes opencv-python &&   apt-get update -y &&   apt-get install -y iputils-ping libgtk2.0-dev
# NOTE: By default CI built wheel packages turn WITH_DISTRIBUTE=OFF,
#       so we must build one with distribute support to install in this image.
RUN pip install paddlepaddle
RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.cifar.train10()" | python'
RUN pip uninstall -y paddlepaddle

# below lines may change a lot for debugging
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/paddle_k8s /usr/bin
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/k8s_tools.py /root
ADD *.whl /
RUN pip install /*.whl && rm -f /*.whl && \
chmod +x /usr/bin/paddle_k8s
ENV LD_LIBRARY_PATH=/usr/local/lib
ADD vgg16_fluid.py vgg16_v2.py /workspace/
