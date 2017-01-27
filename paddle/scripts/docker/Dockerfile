FROM ubuntu:14.04
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_MIRROR
RUN /bin/bash -c 'if [[ -n ${UBUNTU_MIRROR} ]]; then sed -i 's#http://archive.ubuntu.com#${UBUNTU_MIRROR}#g' /etc/apt/sources.list; fi'

RUN apt-get update && \
    apt-get install -y git python-pip python-dev openssh-server bison && \
    apt-get install -y wget unzip tar xz-utils bzip2 gzip coreutils && \
    apt-get install -y curl sed grep graphviz libjpeg-dev zlib1g-dev && \
    apt-get install -y python-numpy python-matplotlib gcc g++ gfortran && \
    apt-get install -y automake clang-3.8 llvm-3.8 libclang-3.8-dev && \
    apt-get clean -y

RUN pip install --upgrade pip
RUN pip install 'protobuf==3.1.0.post1' && \
    pip install -U wheel pillow BeautifulSoup && \
    pip install -U docopt PyYAML sphinx && \
    pip install -U sphinx_rtd_theme recommonmark jupyter

RUN curl -sSL https://cmake.org/files/v3.4/cmake-3.4.1.tar.gz | tar -xz && \
    cd cmake-3.4.1 && ./bootstrap && make -j4 && make install && \
    cd .. && rm -rf cmake-3.4.1

ARG WITH_AVX
ARG WITH_DOC
ARG WITH_STYLE_CHECK

ENV WITH_GPU=OFF
ENV WITH_AVX=${WITH_AVX:-ON}
ENV WITH_DOC=${WITH_DOC:-ON}
ENV WITH_STYLE_CHECK=${WITH_STYLE_CHECK:-OFF}

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

# Jupyter Notebook directory.
RUN mkdir /notes/
WORKDIR "/notes"
EXPOSE 8888

RUN mkdir -p /opt/bin
COPY ./paddle/scripts/docker/entrypoint /opt/bin/

CMD ["/opt/bin/entrypoint"]
