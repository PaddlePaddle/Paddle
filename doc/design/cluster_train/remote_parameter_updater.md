# Design Doc: Remote Parameter Updater for Cluster Train

For an overview of distribute training, please refer to [distributed training design doc](README.md). In this design doc, we will discuss the parameter updater that will use parameter server cclient [The Client Library of Parameter Server Design Doc](pserver_client.md) to manage and update parameters.

## Parameter Updater

Parameter Updater is used by trainer to manage and update parameter, there are mainly two kind of parameter updater: local and remote, since this design is for cluster train, we will only discuss remote parameter updater here.

### Remote Parameter Updater

Remote Parameter Updater manage parameters through remote parameter server with the client that communicate with pserver([The Client Library of Parameter Server Design Doc](pserver_client.md))

In PaddlePaddle Python V2 API, trainer is implemented in python, and the trainer will hold a instance of parameter updater and call it's functions directly. In this design, we will also expose the api of RemoteParameterUpdater to python with swig.

#### Sparse Remote Parameter Updater

Since we will only implement dense parameter management new, the mechanism for sparse parameter will be discussed in next stage.

### Interface Design

TBD
