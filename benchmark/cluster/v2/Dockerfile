FROM paddlepaddle/paddlecloud-job
RUN mkdir -p /workspace && mkdir -p /root/.cache/paddle/dataset/flowers/
ADD vgg16.py reader.py /workspace/
COPY 102flowers.tgz imagelabels.mat setid.mat /root/.cache/paddle/dataset/flowers/

