FROM paddlepaddle/paddlecloud-job
RUN mkdir -p /workspace
ADD reader.py /workspace/
RUN python /workspace/reader.py
ADD vgg16.py /workspace/
