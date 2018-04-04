# Channel Design

## Introduction

A Channel is a data structure that allows for synchronous interprocess
communication via message passing.  It is a fundemental component of CSP
(communicating sequential processes), and allows for users to pass data
between threads without having to worry about synchronization.

## How to use it

Paddle offers python APIs to open and close channels, along with sending
and receiving data to/from a channel.

### Create a channel

Creates a new channel that takes in variables of a specific dtype.

- **fluid.make_channel(dtype, capacity=0)**
  - **dtype**: The data type of variables being sent/received through channel
  - **capacity**: The capacity of the channel.  A capacity of 0 represents
    an unbuffered channel.  Capacity > 0 represents a buffered channel

```
ch = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR, 10)
```

### Close a channel

Closes a channel.  Any pending senders and receivers will be awoken during
this time.  Receivers can still receive from a closed channel, but senders
are not allowed to send any additional data to the channel (Paddle will
raise an exception if users try to send to a closed channel.)

- **fluid.channel_close(channel)**

```
fluid.channel_close(ch)
```

### Send data to a channel

Sends a variable to a channel.  Currently, variables of dtype `LoDTensor`,
`LoDRankTable`, `LoDTensorArray`, `SelectedRows`, `ReaderHolder`, and
`ChannelHolder` are supported.

By default, the data of the Variable is moved from the sender to the receiver,
however the user can optionally copy the data before performing the send.

- **channel_send(channel, variable, is_copy=False)**
  - **channel**: The channel to send the variable to
  - **variable**: The variable to send to the channel
  - **is_copy**: If set to True, channel_send will perform a variable assign
  to copy the source variable to a new variable to be sent.

```
ch = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)
var = fill_constant(shape=[1],dtype=core.VarDesc.VarType.INT32, value=100)
fluid.channel_send(ch, var, True)
```

### Receive data from a channel

Receives a variable from a channel.  The data of the variable is moved to the
receiving variable.

- **channel_recv(channel, return_variable)**
  - **channel**: The channel to receive the variable from
  - **return_variable**: The destination variable used to store the data of the
  variable received from the channel

```
ch = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)
var = fill_constant(shape=[1],dtype=core.VarDesc.VarType.INT32, value=-1)
fluid.channel_recv(ch, var)
```

## How it Works

Channels provides a simple interface for different threads to share data.
To support the synchronization requirements, channels utilizes a series of
internal queues, locks, and conditional variables.

### QueueMessage

QueueMessage encapsulates the state of the channel send/receive operation to be
put in the **sendq/recvq**.  It contains a condition variable used to lock the
thread (when there are no available sends/receives).  In addition, it contains
a callback function to notify a thread when the QueueMessage is being
processed by the channel.

### Queues

- **buff_**: This queue holds the data buffer in a buffered channel.  The
capacity is set to the capacity of the channel.  This data buffer is not
used in an unbuffered channel.

- **sendq**: This queue holds the QueueMessage of any pending senders of a
channel.  When a thread performs a channel_send operation on the channel, the
channel_send operation will put a new QueueMessage on the sendq and block the
current thread under two conditions:
  1. The channel is buffered and is full
  2. The channel is unbuffered and does not have a receiver

- **recvq**:  This queue holds the QueueMessage of any pending receivers of a
channel.  When a thread performs a channel_recv operation on the channel, the
channel_recv operation will put a new QueueMessage on the recvq and block the
current thread under two conditions:
  1. The channel is buffered and there is no data on the buff_
  2. The channel is unbuffered and does not have a sender

### State diagram

#### Channel Send

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/channel_send.png"/><br/>
</p>

#### Channel Receive

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/channel_recv.png"/><br/>
</p>

## Limitations and Considerations

### Variable Copy

In golang, variables in channels are copied from the sender to the receiver.
In Paddle, the data from our variables are **moved** from sender to receiver.
As a result, these variables should not be used after they are sent.  We
provide a flag in channel_send method to allow users to copy the variable to
be sent before it is sent.  

Please note that this is acheived by adding an **assign** operator and creating
a temporary variable that is sent in place of the original variable.  Please
note that **assign** operator has limited support for only certain variables
datatypes.
