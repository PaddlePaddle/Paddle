FROM python:2.7.13-slim
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

# ENV variables
ARG WITH_AVX
ARG WITH_DOC
ARG WITH_STYLE_CHECK

ENV WITH_GPU=OFF
ENV WITH_AVX=${WITH_AVX:-ON}
ENV WITH_DOC=${WITH_DOC:-OFF}
ENV WITH_STYLE_CHECK=${WITH_STYLE_CHECK:-OFF}

ENV HOME /root
ENV LANG en_US.UTF-8

# Use Fix locales to en_US.UTF-8

RUN sed 's@http:\/\/archive.ubuntu.com\/ubuntu\/@mirror:\/\/mirrors.ubuntu.com\/mirrors.txt@' -i /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y libgfortran3 && \
    apt-get clean -y && \
    pip install --upgrade pip && \
    pip install -U 'protobuf==3.1.0'
RUN pip install numpy

ADD build/paddle/trainer/paddle_trainer \
    build/paddle/pserver/paddle_pserver_main \
    build/paddle/trainer/paddle_merge_model \
    /usr/local/opt/paddle/bin/
ADD build/paddle/scripts/usage.sh /usr/local/opt/paddle/bin/paddle_usage
ADD paddle/scripts/submit_local.sh.in /usr/bin/paddle

ADD paddle/dist/py_paddle*linux*.whl build/python/dist/paddle*.whl \
 /usr/local/opt/paddle/share/wheels/
RUN pip install /usr/local/opt/paddle/share/wheels/py_paddle*linux*.whl && \
    pip install /usr/local/opt/paddle/share/wheels/paddle*.whl && \
    rm /usr/local/opt/paddle/share/wheels/py_paddle*linux*.whl && \
    rm /usr/local/opt/paddle/share/wheels/paddle*.whl
ENV PATH="/usr/local/opt/paddle/bin/:${PATH}"
# default command shows the paddle version and exit
CMD ["paddle", "version"]
