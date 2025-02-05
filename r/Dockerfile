FROM ubuntu:18.04
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

WORKDIR /workspace

ENV PATH /opt/python3/bin:/root/.local/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/python3/lib

# Install Python
ADD https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /opt/python3/ && \
    rm -f miniconda3.sh

RUN mkdir -p ~/.pip && \
    echo "[global]" >> ~/.pip/pip.conf && \
    echo "trusted-host =  mirrors.aliyun.com" >> ~/.pip/pip.conf && \
    echo "index-url = https://mirrors.aliyun.com/pypi/simple" >> ~/.pip/pip.conf

RUN echo "channels:" >> ~/.condarc && \
    echo "  - conda-forge" >> ~/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/" >> ~/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/" >> ~/.condarc && \
    echo "  - defaults" >> ~/.condarc && \
    echo "custom_channels:" >> ~/.condarc && \
    echo "  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "show_channel_urls: true" >> ~/.condarc && \
    echo "channel_priority: strict" >> ~/.condarc

# Install R
RUN conda install -y r -c conda-forge

# Install PaddlePaddle
RUN /opt/python3/bin/python -m pip install \
    https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-mkl/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl

# Install reticulate, R interface to Python
RUN Rscript -e 'install.packages("reticulate", repos="https://cran.rstudio.com")'

COPY example example
RUN cd example && \
    curl -O https://paddle-inference-dist.cdn.bcebos.com/mobilenet-test-model-data.tar.gz && \
    tar -zxvf mobilenet-test-model-data.tar.gz && rm mobilenet-test-model-data.tar.gz
