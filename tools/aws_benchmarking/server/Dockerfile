FROM python:2.7.14-stretch

ENV HOME /root
COPY ./ /root/
WORKDIR /root
RUN pip install -r /root/requirements.txt
ENTRYPOINT ["python", "cluster_master.py"]