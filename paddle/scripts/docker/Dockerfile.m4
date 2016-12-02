FROM PADDLE_BASE_IMAGE
MAINTAINER PaddlePaddle Dev Team <paddle-dev@baidu.com>

# It is good to run apt-get install with Dockerfile RUN directive,
# because if the following invocation to /root/build.sh fails, `docker
# build` wouldn't have to re-install packages after we fix
# /root/build.sh.  For more about Docker build cache, please refer to
# https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/#/build-cache.
RUN apt-get update && \
    apt-get install -y cmake libprotobuf-dev protobuf-compiler git \
    libgoogle-glog-dev libgflags-dev libatlas-dev libatlas3-base g++ m4 python-pip \
    python-protobuf python-numpy python-dev swig openssh-server \
    wget unzip python-matplotlib tar xz-utils bzip2 gzip coreutils \
    sed grep graphviz libjpeg-dev zlib1g-dev doxygen && \
    apt-get clean -y
RUN pip install BeautifulSoup docopt PyYAML pillow \
    'sphinx>=1.4.0' sphinx_rtd_theme breathe recommonmark

ENV WITH_GPU=PADDLE_WITH_GPU
ENV WITH_AVX=PADDLE_WITH_AVX

RUN mkdir /paddle
COPY . /paddle/
COPY paddle/scripts/docker/build.sh /root/
RUN /root/build.sh

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
