package pserver

import (
	"errors"
	"fmt"
	"sync"
)

// ElementType is the type of elements of a Parameter.
type ElementType int

var ErrAlreadyInitialized = errors.New("pserver already initialized")

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

type Service struct {
	initialized chan struct{}

	mu       sync.Mutex
	opt      *optimizer
	paramMap map[string]Parameter
}

func NewService() *Service {
	s := &Service{}
	s.paramMap = make(map[string]Parameter)
	s.initialized = make(chan struct{})
	return s
}

func (s *Service) BeginInitParams(config []byte, dummy *int) error {
	select {
	case <-s.initialized:
		return ErrAlreadyInitialized
	default:
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.opt != nil {
		s.opt.Cleanup()
	}

	// TODO(helin): parse learning rate from config
	s.opt = newOptimizer(sgd, 0.01)
	return nil
}

func (s *Service) InitParam(paramWithConfigs ParameterWithConfig, dummy *int) error {
	select {
	case <-s.initialized:
		return ErrAlreadyInitialized
	default:
	}

	// TODO(helin): parse parameter config

	s.mu.Lock()
	defer s.mu.Unlock()

	// TODO(helin): check if paramWithConfigs.Param.Content is
	// properly memory aligned, if not, make copy to a memory
	// aligned region.
	s.paramMap[paramWithConfigs.Param.Name] = paramWithConfigs.Param
	return nil
}

func (s *Service) FinishInitParams(dummy0 int, dummy1 *int) error {
	select {
	case <-s.initialized:
		return ErrAlreadyInitialized
	default:
	}

	close(s.initialized)
	return nil
}

func (s *Service) SendGrads(grads []Gradient, dummy *int) error {
	<-s.initialized

	count := len(grads)
	if count == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for _, g := range grads {
		if _, ok := s.paramMap[g.Name]; !ok {
			return fmt.Errorf("parameter: %s does not exist", g.Name)
		}
	}

	errCh := make(chan error, count)
	for _, g := range grads {
		go func(p Parameter, g Gradient) {
			err := s.opt.UpdateParameter(p, g)
			errCh <- err
		}(s.paramMap[g.Name], g)
	}

	recv := 0
	for err := range errCh {
		if err != nil {
			return err
		}

		recv++
		if recv == count {
			break
		}
	}
	return nil
}

func (s *Service) GetParams(names []string, parameters *[]Parameter) error {
	<-s.initialized
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, n := range names {
		if _, ok := s.paramMap[n]; !ok {
			return fmt.Errorf("parameter: %s does not exist", n)
		}
	}

	*parameters = make([]Parameter, len(names))
	for i, n := range names {
		// The parameter content (a byte slice) may change
		// during RPC serialization due to write from other
		// goroutine, we allow it since mini-batch based deep
		// learning optimization methods are stochastic in
		// nature. This race condition is allowed deliberately
		// to save the program from making a copy of the
		// paramter content.
		(*parameters)[i] = s.paramMap[n]
	}

	return nil
}

func (s *Service) SaveModel(path string, dummy *int) error {
	<-s.initialized

	// TODO
	return nil
}
