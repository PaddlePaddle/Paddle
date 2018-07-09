FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Use UBUNTU_MIRROR can speed up apt-get speed.
# ARG UBUNTU_MIRROR
# RUN /bin/bash -c 'if [[ -n ${UBUNTU_MIRROR} ]]; then sed -i 's#http://archive.ubuntu.com/ubuntu#${UBUNTU_MIRROR}#g' /etc/apt/sources.list; fi'

RUN apt-get update && apt-get install -y python python-pip iputils-ping libgtk2.0-dev wget vim net-tools iftop python-opencv
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/lib/libcudnn.so && ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/libnccl.so

# IMPORTANT:
# Add "ENV http_proxy=http://ip:port" if your download is slow, and don't forget to unset it at runtime.
# exmaple: unset http_proxy && unset https_proxy && python fluid_benchmark.py ...

RUN pip install -U pip
RUN pip install -U kubernetes paddlepaddle

RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.cifar.train10()\npaddle.dataset.flowers.fetch()" | python'
RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.mnist.train()\npaddle.dataset.mnist.test()\npaddle.dataset.imdb.fetch()" | python'
RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.imikolov.fetch()" | python'
RUN pip uninstall -y paddlepaddle && mkdir /workspace

ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/paddle_k8s /usr/bin
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/k8s_tools.py /root
RUN chmod +x /usr/bin/paddle_k8s

ADD *.whl /
RUN pip install /*.whl && rm -f /*.whl 

ENV LD_LIBRARY_PATH=/usr/local/lib
ADD fluid_benchmark.py recordio_converter.py args.py recordio_converter.py run.sh run_fluid_benchmark.sh /workspace/
ADD models/ /workspace/models/
