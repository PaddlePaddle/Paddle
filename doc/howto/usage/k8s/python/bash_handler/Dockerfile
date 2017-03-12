FROM ubuntu:14.04
RUN sudo sed 's@http:\/\/archive.ubuntu.com\/ubuntu\/@mirror:\/\/mirrors.ubuntu.com\/mirrors.txt@' -i /etc/apt/sources.list  && \
    apt-get update && \
    apt-get install -y wget unzip tar curl sed git && \
    apt-get clean -y
    
CMD ["/bin/bash"]
