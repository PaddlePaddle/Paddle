#FROM paddlepaddle/paddlecloud-job
#RUN mkdir -p /workspace
#ADD reader.py /workspace/
#RUN python /workspace/reader.py
FROM python:2.7.14
ADD *.whl /
RUN pip install /*.whl && rm -f /*.whl
ADD paddle_k8s /usr/bin
ADD k8s_tools.py /root
RUN pip install -U kubernetes opencv-python &&   apt-get update -y &&   apt-get install -y iputils-ping libgtk2.0-dev 

ADD vgg16.py /workspace/
