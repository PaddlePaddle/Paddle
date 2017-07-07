package pserver

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	log "github.com/sirupsen/logrus"
)

// ElementType is the type of elements of a Parameter.
type ElementType int

const (
	// AlreadyInitialized is true if pserver is initialized
	AlreadyInitialized = "pserver already initialized"
	// Uninitialized is true if pserver not fully initialized
	Uninitialized = "pserver not fully initialized"
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

// Checkpoint of Parameter and State
type parameterCheckPoint struct {
	ParamConfig ParameterWithConfig
	State       []byte
}

// checkpoint signature
type checkpointMeta struct {
	UUID      string `json:"uuid"`
	Md5sum    string `json:"md5sum"`
	Timestamp string `json:"timestamp"`
}

// Checkpoint is the pserver shard persist in file
type Checkpoint []parameterCheckPoint

// Gradient is the gradient of the parameter.

// Service is the RPC service for pserver.
type Service struct {
	initialized        chan struct{}
	idx                int
	checkpointInterval int
	checkpointPath     string
	client             *EtcdClient
	mu                 sync.Mutex
	optMap             map[string]*optimizer
}

// //serialize ParameterWithConfig to byte stream
// func GetBytes(content ...interface{}) ([]byte, error) {

// 	var buf bytes.Buffer
// 	encoder := gob.NewEncoder(&buf)
// 	err := encoder.Encode(content)
// 	if err != nil {
// 		return nil, err
// 	}
// 	return buf.Bytes(), nil
// }

// NewService creates a new service, will bypass etcd registration if no
// endpoints specified.
func NewService(idx int, seconds int, path string, client *EtcdClient, cp Checkpoint) (*Service, error) {
	s := &Service{
		idx:                idx,
		checkpointInterval: time.Second * time.Duration(seconds),
		checkpointPath:     path,
		client:             client,
	}
	s.optMap = make(map[string]*optimizer)
	s.initialized = make(chan struct{})

	if cp != nil {
		for _, item := range cp {
			p := item.ParamConfig
			st := item.State
			s.optMap[p.Param.Name] = newOptimizer(p, st)
		}
	}
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

// pserver save checkpoint
func (s *Service) doCheckpoint() error {
	<-s.initialized
	s.mu.Lock()
	defer s.mu.Unlock()

	cp := make([]parameterCheckPoint, 0, len(s.optMap))
	index := 0
	for name, opt := range s.optMap {
		var pc parameterCheckPoint
		pc.ParamConfig.Param.Name = name
		pc.ParamConfig.Param.ElementType = opt.elementType
		pc.ParamConfig.Param.Content = opt.GetWeights()
		pc.State = opt.GetStates()
		cp[index] = pc
		index++
	}
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(cp)
	if err != nil {
		return err
	}

	cpMeta := checkpointMeta{}
	cpMeta.UUID = s.checkpointPath + strconv.Itoa(s.idx)
	cpMeta.Timestamp = time.Now().String()
	h := md5.New()
	cpMeta.Md5sum = h.Sum(buf.Bytes())

	cpMetajson, err := json.Marshal(cpMeta)
	s.client.PutKey(filepath.Join(PsCheckpoint, strconv.Itoa(s.idx)), cpMetajson, 3)
	if err != nil {
		return err
	}
	if _, err = os.Stat(cpMeta.UUID); os.IsNotExist(err) {
		log.Info("checkpoint does not exists.")
	} else {
		err = os.Remove(cpMeta.UUID)
		log.Infof("checkpoint %s already exsits, removing ", cpMeta.UUID)
	}
	f, err := os.Create(cpMeta.UUID)
	defer f.Close()
	if err != nil {
		log.Errorln(err)
	}
	writer := bufio.NewWriter(f)
	_, err = writer.Write(buf.Bytes())
	writer.Flush()
	if err != nil {
		log.Errorln(err)
	}
	return nil
}
