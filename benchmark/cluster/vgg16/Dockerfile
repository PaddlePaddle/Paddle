FROM python:2.7.14
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/paddle_k8s /usr/bin
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/k8s_tools.py /root
RUN pip install -U kubernetes opencv-python &&   apt-get update -y &&   apt-get install -y iputils-ping libgtk2.0-dev && \
chmod +x /usr/bin/paddle_k8s
# NOTE: By default CI built wheel packages turn WITH_DISTRIBUTE=OFF,
#       so we must build one with distribute support to install in this image.
ADD *.whl /
RUN pip install /*.whl && rm -f /*.whl
ENV LD_LIBRARY_PATH=/usr/local/lib
RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.cifar.train10()" | python'

ADD vgg16_fluid.py vgg16_v2.py /workspace/
