FROM yancey1989/paddle:0.10.0rc4
RUN pip install kubernetes
ADD ./paddle_k8s /usr/bin
ADD ./k8s_tools.py /root/
CMD ["paddle_k8s"]
