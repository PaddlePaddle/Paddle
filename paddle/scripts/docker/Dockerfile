FROM ubuntu:14.04
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

RUN apt-get update \
    && apt-get install -y cmake libprotobuf-dev protobuf-compiler git \
    libgoogle-glog-dev libgflags-dev libatlas-dev libatlas3-base g++ m4 python-pip \
    python-protobuf python-numpy python-dev swig openssh-server \
    wget unzip python-matplotlib tar xz-utils bzip2 gzip coreutils \
    sed grep graphviz libjpeg-dev zlib1g-dev doxygen \
    clang-3.8 llvm-3.8 libclang-3.8-dev \
    && apt-get clean -y
RUN pip install -U BeautifulSoup docopt PyYAML pillow \
    sphinx sphinx_rtd_theme breathe recommonmark

ARG WITH_AVX
ENV WITH_AVX=${WITH_AVX:-ON}
ENV WITH_GPU=OFF

RUN mkdir /paddle
COPY . /paddle/
RUN /paddle/paddle/scripts/docker/build.sh
VOLUME ["/usr/share/nginx/html/data", "/usr/share/nginx/html/paddle"]

RUN echo 'export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH}' >> /etc/profile
RUN pip install /usr/local/opt/paddle/share/wheels/*.whl
RUN paddle version  # print version after build

# Configure OpenSSH server. c.f. https://docs.docker.com/engine/examples/running_ssh_service
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
