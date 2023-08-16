- Dockerfile.conda.musa: use minconda to manage python environment
- Dockerfile.musa: consistent with paddle official, py37, py38 and py39 were installed at `/usr/local/bin` by default

You can comment out the last few lines in the Dockerfile, i.e.
```shell
WORKDIR /home
COPY ./paddle_musa paddle_musa
RUN cd paddle_musa && \
    pre-commit install && \
    pre-commit run
```
and use `-v` to map the local `paddle_musa` into the container when creating the docker container (we keep this only for the convenience of CI).

Then, you can run these commands below to create a new paddle_musa docker image and run it:
```shell
# DOCKER_FILE=/path/to/Dockerfile
bash build.sh -i paddle_musa_docker    \
              -m ${MUSA_TOOLKITS_URL}  \
              -n ${MUDNN_URL}          \
              -f ${DOCKER_FILE}

docker run -it  --privileged=true                    \
                --env MTHREADS_VISIBLE_DEVICES=all   \
                --name paddle_musa_example           \
                --shm-size 80G ${IMAGE_NAME} /bin/bash
```
