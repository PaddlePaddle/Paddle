FROM python:3.7-alpine3.10

WORKDIR /root

RUN apk update

RUN apk add --no-cache \
    g++  gfortran make cmake patchelf git \
    linux-headers \
    freetype-dev libjpeg-turbo-dev zlib-dev

RUN apk add --no-cache --force-overwrite \
    lapack-dev openblas-dev 

ENTRYPOINT [ "/bin/sh" ]
