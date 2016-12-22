FROM paddledev/paddle:cpu-devel-latest
COPY build.sh /
RUN pip install sphinx &&\
    pip install sphinx_rtd_theme &&\
    apt install -y doxygen graphviz &&\
    pip install recommonmark numpy protobuf==2.6.1
CMD /build.sh
