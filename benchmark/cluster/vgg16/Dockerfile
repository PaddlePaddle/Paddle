FROM paddlepaddle/paddle:latest-dev
RUN pip install -U kubernetes opencv-python &&   apt-get update -y &&   apt-get install -y iputils-ping libgtk2.0-dev

# NOTE: By default CI built wheel packages turn WITH_DISTRIBUTE=OFF,
#       so we must build one with distribute support to install in this image.
ADD *.whl /
RUN pip install /*.whl && rm -f /*.whl
# if download slowly, you can add proxy here.
# ENV https_proxy=
RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.cifar.train10()" | python'
# unset proxy if it is setted.
# ENV https_proxy=""

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
