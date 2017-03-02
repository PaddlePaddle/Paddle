FROM paddledev/paddle:cpu-devel-latest
COPY build.sh /
RUN pip install sphinx==1.5.3 &&\
    pip install sphinx_rtd_theme==0.1.10b0 &&\
    apt install -y doxygen graphviz &&\
    pip install recommonmark numpy protobuf==2.6.1
CMD /build.sh
