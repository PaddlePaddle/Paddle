FROM registry.baidu.com/paddlepaddle/rawjob
RUN mkdir -p /workspace && mkdir -p /root/.cache/paddle/dataset/flowers/
ADD vgg16.py reader.py /workspace/
ADD 102flowers.tgz imagelabels.mat setid.mat /root/.cache/paddle/dataset/flowers/
