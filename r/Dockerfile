FROM ubuntu:18.04
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

WORKDIR /workspace

ENV PATH /opt/python3/bin:/root/.local/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/python3/lib

# Install Python
ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /opt/python3/ && \
    rm -f miniconda3.sh

RUN mkdir -p ~/.pip && \
    echo -e "[global]" > ~/.pip/pip.conf && \
    echo -e "trusted-host =  mirrors.aliyun.com" >> ~/.pip/pip.conf && \
    echo -e "index-url = https://mirrors.aliyun.com/pypi/simple" >> ~/.pip/pip.conf

RUN echo "channels:" >> ~/.condarc \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/" >> ~/.condarc \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/" >> ~/.condarc \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/" >> ~/.condarc \
    echo "  - defaults" >> ~/.condarc \
    echo "show_channel_urls: true" >> ~/.condarc
    
# Install R
RUN conda install -y r

# Install PaddlePaddle
RUN /opt/python3/bin/python -m pip install -y paddlepaddle

# Install reticulate, R interface to Python
RUN Rscript -e 'install.packages("reticulate", repos="https://cran.rstudio.com")'

