## Design Doc : Trainer 

the trainer role in whole system please refer to [distributed training design doc](./README.md). 

This design doc only focus on Master Server and Trainer synchronization and python client event processing . The Task dispatch interface please refer to the [master_server](./master_server.md), [data_dispatch](./data_dispatch.md) and so on.

## Synchronize SGD

In synchronize SGD, trainer need to wait other nodes finish training in every minibatch. And don't go on next epoch training if there is any node lag behind.

To wait other trainer in same training minibatch, the trainer call get_params will be blocked until pserver had finished the model update.

<img src="src/paddle-trainer.png" width="600"/>

To wait other trainer in same epoch, use the waitEpochFinish to decide if an epoch has finished and enter next training epoch.

```go
// Master Service 
func(s *Service) waitEpochFinish(dummy int, epoch_id *int) error;
```

## Event Handler

To select the trainer for process Python client event,  same way as initialization parameters. Every trainer will try to get a distribute lock, then election a leader one. Leader trainer will keep to writing a file/ send metric data to evaluatorServer. Then python client can use that data draw metrics in real time.
