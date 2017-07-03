package pserver

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"encoding/gob"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"

	log "github.com/sirupsen/logrus"
)

// ElementType is the type of elements of a Parameter.
type ElementType int

const (
	AlreadyInitialized = "pserver already initialized"
	Uninitialized      = "pserver not fully initialized"
)

const (
	checkpoint_path = "/checkpoints/"
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

// PsDesired is etcd path for store desired pserver count
const PsDesired = "/ps_desired"

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
	State  []byte // parameter training state
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

type Checkpoint struct {
	uuid      string
	md5sum    string
	timestamp string
}

//serialize ParameterWithConfig to byte stream
func GetBytes(content ...interface{}) ([]byte, error) {

	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(content)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
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
	//FIXME: checkpoint is only used by pserver
	// and has a constant path of */checkpoints/{pserver_idx}*
	<-s.initialized
	s.mu.Lock()
	defer s.mu.Unlock()
	var paramWithConfig ParameterWithConfig
	for name, opt := range s.optMap {
		paramWithConfig.Param.Name = name
		paramWithConfig.Param.ElementType = opt.elementType
		paramWithConfig.Param.Content = opt.GetWeights()
		paramWithConfig.State = opt.GetStates()
		content, err := GetBytes(paramWithConfig)
		if err != nil {
			log.Errorln(err)
		}
		ck := Checkpoint{}
		h := md5.New()
		ck.md5sum = hex.EncodeToString(h.Sum(content))
		ck.timestamp = time.Now().String()
		ck.uuid = checkpoint_path + strconv.Itoa(s.idx)
		ckbytes, err := GetBytes(ck)
		if err != nil {
			log.Errorln(err)
		}
		// TODO: according design doc, need to save uuid to etcd in json format
		// {\"uuid\": [UUID], \"md5\", \"MD5 sum\", \"timestamp\": xxxx}
		log.Infof("parameter checkpoint %s", ckbytes)

		if _, err = os.Stat(ck.uuid); os.IsNotExist(err) {
			log.Info("checkpoint not exists.")
		} else {
			err = os.Remove(ck.uuid)
			log.Infof("remove %s", ck.uuid)
		}
		f, err := os.Create(ck.uuid)
		defer f.Close()
		if err != nil {
			log.Errorln(err)
		}
		writer := bufio.NewWriter(f)
		_, err = writer.Write(content)
		if err != nil {
			log.Errorln(err)
		}
	}
	return nil
}
