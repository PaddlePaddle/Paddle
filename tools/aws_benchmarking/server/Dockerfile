# A image for building paddle binaries
# Use cuda devel base image for both cpu and gpu environment
FROM python:2.7.14-stretch

ENV HOME /root
# Add bash enhancements
COPY ./ /root/
WORKDIR /root
RUN pip install -r /root/requirements.txt
ENTRYPOINT ["python", "cluster_master.py"]