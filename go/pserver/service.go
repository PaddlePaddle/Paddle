// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pserver

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"encoding/gob"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	log "github.com/sirupsen/logrus"
)

// ElementType is the type of elements of a Parameter.
type ElementType int

// RPC error message.
const (
	AlreadyInitialized  = "pserver already initialized"
	Uninitialized       = "pserver not fully initialized"
	CheckpointMD5Failed = "checkpoint file MD5 validation failed"
)

// Supported element types.
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

// checkpointMeta saves checkpoint metadata
type checkpointMeta struct {
	UUID      string `json:"uuid"`
	MD5       string `json:"md5"`
	Timestamp int64  `json:"timestamp"`
}

// Checkpoint is the pserver shard persist in file
type Checkpoint []parameterCheckpoint

// Gradient is the gradient of the parameter.
type Gradient Parameter

// Service is the RPC service for pserver.
type Service struct {
	initialized        chan struct{}
	idx                int
	checkpointInterval time.Duration
	checkpointPath     string
	client             *EtcdClient
	mu                 sync.Mutex
	optMap             map[string]*optimizer
}

// parameterCheckpoint saves parameter checkpoint
type parameterCheckpoint struct {
	ParameterWithConfig
	State []byte
}

// NewCheckpointFromFile loads parameters and state from checkpoint file
func NewCheckpointFromFile(cpPath string, idx int, e *EtcdClient) (Checkpoint, error) {
	v, err := e.GetKey(PsPath+string(idx), 3*time.Second)
	if err != nil {
		return nil, err
	}

	var cpMeta checkpointMeta
	if err = json.Unmarshal(v, &cpMeta); err != nil {
		return nil, err
	}

	fn := filepath.Join(cpPath, cpMeta.UUID)
	if _, err = os.Stat(fn); os.IsNotExist(err) {
		return nil, err
	}
	content, err := ioutil.ReadFile(fn)
	if err != nil {
		return nil, err
	}

	h := md5.New()
	md5 := hex.EncodeToString(h.Sum(content))
	if md5 != cpMeta.MD5 {
		return nil, errors.New(CheckpointMD5Failed)
	}

	dec := gob.NewDecoder(bytes.NewReader(content))
	cp := Checkpoint{}
	if err = dec.Decode(cp); err != nil {
		return nil, err
	}
	return cp, nil
}

// NewService creates a new service, will bypass etcd registration if no
// endpoints specified. It will recovery from checkpoint file if a exists a specified checkpoint.
func NewService(idx int, interval time.Duration, path string, client *EtcdClient, cp Checkpoint) (*Service, error) {
	s := &Service{
		idx:                idx,
		checkpointInterval: interval,
		checkpointPath:     path,
		client:             client,
	}
	s.optMap = make(map[string]*optimizer)
	s.initialized = make(chan struct{})

	if cp != nil {
		for _, item := range cp {
			p := ParameterWithConfig{
				Param:  item.Param,
				Config: item.Config,
			}
			s.optMap[p.Param.Name] = newOptimizer(p, item.State)
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
	s.optMap[paramWithConfigs.Param.Name] = newOptimizer(paramWithConfigs, nil)
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
	// parameter content.
	parameter.Name = name
	parameter.ElementType = opt.elementType
	parameter.Content = opt.GetWeights()
	return nil
}

// pserver save checkpoint
func (s *Service) doCheckpoint() (err error) {
	<-s.initialized
	s.mu.Lock()
	defer s.mu.Unlock()

	cp := make([]parameterCheckpoint, len(s.optMap))
	index := 0
	for name, opt := range s.optMap {
		var pc parameterCheckpoint
		pc.Param.Name = name
		pc.Param.ElementType = opt.elementType
		pc.Param.Content = opt.GetWeights()
		pc.State = opt.GetStates()
		cp[index] = pc
		index++
	}
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err = encoder.Encode(cp)
	if err != nil {
		return
	}

	cpMeta := checkpointMeta{}
	cpMeta.UUID = s.checkpointPath + strconv.Itoa(s.idx)
	cpMeta.Timestamp = time.Now().UnixNano()
	h := md5.New()
	cpMeta.MD5 = hex.EncodeToString(h.Sum(buf.Bytes()))

	cpMetajson, err := json.Marshal(cpMeta)
	if err != nil {
		return
	}

	err = s.client.PutKey(filepath.Join(PsCheckpoint, strconv.Itoa(s.idx)), cpMetajson, 3*time.Second)
	if err != nil {
		return
	}
	if _, err = os.Stat(cpMeta.UUID); os.IsNotExist(err) {
		log.Info("checkpoint does not exists.")
	} else {
		err = os.Remove(cpMeta.UUID)
		if err != nil {
			log.Infof("Removing checkpoint %s failed", cpMeta.UUID)
		} else {
			log.Infof("checkpoint %s already exsits, removing ", cpMeta.UUID)
		}
	}
	f, err := os.Create(cpMeta.UUID)
	if err != nil {
		return
	}

	defer func() {
		closeErr := f.Close()
		if closeErr != nil {
			if err != nil {
				log.Errorln(closeErr)
			} else {
				// Set closeErr as return value.
				err = closeErr
			}
		}
	}()

	writer := bufio.NewWriter(f)
	_, err = writer.Write(buf.Bytes())
	if err != nil {
		return
	}

	err = writer.Flush()
	if err != nil {
		return
	}

	return
}
