import yi_json

g = 100
def read():
    queue q;
    # warmup q
    for i = 0 : 1000
        q.push(read())
    yield q.shuffle_get()

input = paddle.layer.data(...)
intermediate = paddle.layers.fc(input)
output = paddle.layer.softmax(intermediate)

model = paddle.model.create(output)

train(model, data_provider=read, cluster="clusterId")

#--------------------------------------------------------------------------------

# 1. package, docker build, docker push
# 2. kubectl, clusterId Kuberentes job, 10 trainer containers, 5 parameter server containers

#--------------------------------------------------------------------------------

def train():
    if os.environ["kube_api_server"] == nil:
        docker_build()
        docker_push()
        kube_ctrl_start_job()
    else:
        rank = kube_mpi_rank()
        if rank == 0:
            master()
        elif rank >= 15:
            parameter_server()
        else:
            _train()
