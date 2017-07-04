package pserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
)

// ElementType is the type of elements of a Parameter.
type ElementType int

const (
	AlreadyInitialized = "pserver already initialized"
	Uninitialized      = "pserver not fully initialized"
)

// Supported element types
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
	Content     []byte
}

// ParameterWithConfig contains the parameter and the configuration.
type ParameterWithConfig struct {
	Param  Parameter
	Config []byte // parameter configuration in Proto Buffer format
}

// Gradient is the gradient of the parameter.
type Gradient Parameter

// Service is the RPC service for pserver.
type Service struct {
	initialized chan struct{}
	idx         int

	mu     sync.Mutex
	optMap map[string]*optimizer
}

// Checkpoint saves the checkpoint for pserver
type Checkpoint struct {
	UUID      string `json:"uuid"`
	MD5       string `json:"md5"`
	Timestamp string `json:"timestamp"`
	State     []byte
	ParameterWithConfig
}

// NewCheckpoint creates a new checkpoint.
func NewCheckpoint(idx int, cpPath string, e *EtcdClient) (*Checkpoint, error) {
	v, err := e.GetCheckpointInfo(idx)
	if err != nil {
		return nil, err
	}
	var cp Checkpoint
	if err = json.Unmarshal(v, &cp); err != nil {
		return nil, err
	}
	fn := filepath.Join(cpPath, cp.UUID)
	if _, err = os.Stat(fn); os.IsNotExist(err) {
		return nil, err
	}

	f, err := os.Open(fn)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	buf, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}
	// TODO: create checkpoint from file

	return nil, nil
}

// NewServiceFromCheckpoint creates a new service with the specified checkpoint
func NewServiceFromCheckpoint(idx int, cp *Checkpoint) (*Service, error) {
	// TODO: create service from checkpoint
	return nil, nil
}

// NewService creates a new service, will bypass etcd registration if no
// endpoints specified.
func NewService(idx int) (*Service, error) {
	s := &Service{
		idx: idx,
	}
	s.optMap = make(map[string]*optimizer)
	s.initialized = make(chan struct{})
	return s, nil
}

// InitParam initializes a parameter.
func (s *Service) InitParam(paramWithConfigs ParameterWithConfig, dummy *int) error {
	select {
	case <-s.initialized:
		return errors.New(AlreadyInitialized)
	default:
	}

	// TODO(helin): parse parameter config

	s.mu.Lock()
	defer s.mu.Unlock()

	// TODO(helin): check if paramWithConfigs.Param.Content is
	// properly memory aligned, if not, make copy to a memory
	// aligned region.
	s.optMap[paramWithConfigs.Param.Name] = newOptimizer(paramWithConfigs)
	return nil
}

// FinishInitParams tells the parameter server that the parameter
// initialization has finished.
func (s *Service) FinishInitParams(dummy0 int, dummy1 *int) error {
	select {
	case <-s.initialized:
		return errors.New(AlreadyInitialized)
	default:
	}

	close(s.initialized)
	return nil
}

// SendGrad sends gradient to parameter servers for parameter
// optimization.
func (s *Service) SendGrad(g Gradient, dummy *int) error {
	select {
	case <-s.initialized:
	default:
		return errors.New(Uninitialized)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	o, ok := s.optMap[g.Name]
	if !ok {
		return fmt.Errorf("parameter: %s does not exist", g.Name)
	}

	return o.UpdateParameter(g)
}

// GetParam gets parameters from the parameter server.
func (s *Service) GetParam(name string, parameter *Parameter) error {
	<-s.initialized
	s.mu.Lock()
	defer s.mu.Unlock()

	opt, ok := s.optMap[name]
	if !ok {
		return fmt.Errorf("parameter: %s does not exist", name)
	}

	// The parameter content (a byte slice) may change
	// during RPC serialization due to write from other
	// goroutine, we allow it since mini-batch based deep
	// learning optimization methods are stochastic in
	// nature. This race condition is allowed deliberately
	// to save the program from making a copy of the
	// paramter content.
	parameter.Name = name
	parameter.ElementType = opt.elementType
	parameter.Content = opt.GetWeights()
	return nil
}

// Save tells the parameter server to save parameters.
func (s *Service) Save(path string, dummy *int) error {
	<-s.initialized

	// TODO
	return nil
}
