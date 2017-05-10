# Design Doc: Trainer Communication Library

For an overview of trainer's role, please refer to [distributed training design doc](README.md). In this design doc, we will discuss the trainer's communication library, which will manage communication with parameter servers and the [master server](master_server.md). The library will be implemented in [Go](https://golang.org/) and made available as a static or dynamic library with a C header file.

## Go Interface

The Go interface is the basic abstraction of communications with the master server and parameter servers. We will add another layer on top (add retry logic, polish interface with C idiom) before exposing the library with a [C interface](#c-interface).

```go
// MasterClient is the client to the master server.
type MasterClient struct {}

// GetTask gets a new task by telling the master server the finished task.
// Use nil as the finished task when getting the task for the first time.
func (*MasterClient) GetTask(finished master.Task) (master.Task, error)

// ElementType is the type of elements of a Parameter.
type ElementType int

// Different element types.
const (
	Int32 ElementType = iota
	UInt32
	Int64
	UInt64
	Float32
	Float64
)

// Parameter is a piece of data to sync with the parameter server.
type Parameter struct {
	Name        string
	ElementType ElementType
	Buffer      []byte
}

// Gradient is the gradient of the parameter.
type Gradient Parameter

// PServerClient is the client to parameter servers.
type PServerClient struct {}

// UpdateRule specifies the rule for updating parameters with gradients.
type UpdateRule struct {
	UpdateMethod pserver.UpdateMethod
	LearningRate float32
}

// ParamInitChans returns a send channel for parameter initialization.
//
// ParamInitChans will be called from multiple trainers, only one trainer should
// initialize the parameters on parameter servers, other trainers will instead
// get the initialized parameters from parameter servers using GetParam.
//
// If send channel is not nil, the trainer is selected to do the initialization,
// the trainer needs to signal for finishing initializing the parameters by
// closing the send channel.
func (*PServerClient) ParamInitChan() (send chan<- Parameter, err error)

// SendGrad sends gradients to parameter servers.
func (*PServerClient) SendGrad(method UpdateMethod, grads []Gradient) error

// GetParam gets parameters from parameter servers.
func (*PServerClient) GetParam(names []string) ([]Parameter, error)

// Save indicates parameters to save the parameter to the given path.
//
// Path needs to be the path to a distributed file system which is visible
// to all parameter servers.
func (*PServerClient) Save(path string) error
```
Please see [master server design doc](master_server.md) for the definition of `master.Task`.

## C Interface

TODO
