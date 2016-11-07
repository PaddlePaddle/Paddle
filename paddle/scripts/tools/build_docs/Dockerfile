FROM paddledev/paddle:cpu-devel-latest
COPY build.sh /
RUN pip install sphinx &&\
    apt install -y doxygen graphviz &&\
    pip install breathe recommonmark numpy protobuf==2.6.1
CMD /build.sh
