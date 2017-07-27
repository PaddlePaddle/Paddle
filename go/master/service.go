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

package master

import (
	"bytes"
	"compress/gzip"
	"encoding/gob"
	"errors"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

	log "github.com/sirupsen/logrus"

	"github.com/PaddlePaddle/recordio"
)

const (
	dialTimeout = 5 * time.Second
)

// ErrAllTaskFailed occur when tasks are in done or failed state.
var ErrAllTaskFailed = errors.New("all task finished")

// ErrNoMoreAvailable occur when no task in todo and yet not all done or fail.
var ErrNoMoreAvailable = errors.New("no more available task")

// ErrPassBefore client side pass number does not match with master counter.
var ErrPassBefore = errors.New("pass number smaller than master")

// ErrPassAfter client side pass number does not match with master counter.
var ErrPassAfter = errors.New("pass number larger than master")

// Store is the interface for save and load the master state.
type Store interface {
	Save([]byte) error
	Load() ([]byte, error)
}

// Chunk is a chunk of data consisted of several data instances.
type Chunk struct {
	Path  string
	Index recordio.Index // chunk index
}

// TaskMeta is a struct which stores task's meta info.
type TaskMeta struct {
	ID    int
	Epoch int
}

// Task is the basic unit of data instances assigned to trainers.
type Task struct {
	Meta   TaskMeta
	Chunks []Chunk
}

type taskEntry struct {
	Task Task
	// A task fails if it's timeout or trainer reports it exits unnormally.
	NumFailure int
}

type taskQueues struct {
	Todo    []taskEntry
	Pending map[int]taskEntry // map from task ID to task entry
	Done    []taskEntry
	Failed  []taskEntry
}

// Service is the master server service.
type Service struct {
	chunksPerTask int
	timeoutDur    time.Duration
	failureMax    int
	store         Store

	ready    chan struct{}
	initDone bool

	mu         sync.Mutex
	taskQueues taskQueues
	currPass   int
	jobTasks   []taskEntry

	savingTrainer string
}

func partition(chunks []Chunk, chunksPerTask int) []taskEntry {
	// generate uniq id across job using nanosecond + randint + counter
	// FIXME(typhoonzero): this is a workaround, use uuid
	randStart := rand.Int()
	counter := 0
	timestamp := time.Now().Nanosecond()
	id := timestamp + randStart + counter
	if chunksPerTask <= 0 {
		chunksPerTask = 1
	}

	var result []taskEntry
	var cur taskEntry
	for i, c := range chunks {
		if i%chunksPerTask == 0 && len(cur.Task.Chunks) > 0 {
			cur.Task.Meta.ID = id
			counter++
			id = timestamp + randStart + counter
			result = append(result, cur)
			cur.Task.Chunks = nil
		}

		cur.Task.Chunks = append(cur.Task.Chunks, c)
	}

	if len(cur.Task.Chunks) > 0 {
		cur.Task.Meta.ID = id
		result = append(result, cur)
	}

	return result
}

// NewService creates a new service.
func NewService(store Store, chunksPerTask int, timeoutDur time.Duration, failureMax int) (*Service, error) {
	s := &Service{}
	s.chunksPerTask = chunksPerTask
	s.timeoutDur = timeoutDur
	s.failureMax = failureMax
	s.taskQueues = taskQueues{}
	s.taskQueues.Pending = make(map[int]taskEntry)
	s.ready = make(chan struct{})
	s.store = store
	recovered, err := s.recover()
	if err != nil {
		return nil, err
	}

	if recovered {
		// Recovered. Now the state is already initialized,
		// and the master is ready.
		s.initDone = true
		close(s.ready)
		log.Info("Master recovered from saved state.")
	}

	return s, nil
}

// recover recovers service state from etcd.
func (s *Service) recover() (bool, error) {
	state, err := s.store.Load()
	if err != nil {
		return false, err
	}

	if state == nil {
		log.Infoln("No state exists, not recovered.")
		return false, nil
	}

	log.Infof("Loaded snapshot of size: %d bytes.", len(state))
	gr, err := gzip.NewReader(bytes.NewReader(state))
	if err != nil {
		return false, err
	}

	dec := gob.NewDecoder(gr)
	var tqs taskQueues
	err = dec.Decode(&tqs)
	if err != nil {
		return false, err
	}

	err = gr.Close()
	if err != nil {
		// Only close failed, recover actually succeed, so
		// just log error.
		log.Errorln(err)
	}

	s.taskQueues = tqs
	return true, nil
}

// snapshot *must* be called with s.mu being held.
func (s *Service) snapshot() error {
	// TODO(helin): etcd request has a size limit, so the snapshot
	// size is limited by the max request size. We should either
	// divide the snapshot into smaller chunks and save under
	// different keys, or configure the request size to be big
	// enough:
	// https://github.com/coreos/etcd/blob/2f84f3d8d8ed8f9537ab6ffa44a3a1c7eddfa9b1/embed/config.go#L44
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	enc := gob.NewEncoder(gw)
	err := enc.Encode(s.taskQueues)
	if err != nil {
		return err
	}
	err = gw.Close()
	if err != nil {
		return err
	}

	state := buf.Bytes()
	log.Infof("Saving snapshot of size: %d bytes.", len(state))
	return s.store.Save(state)
}

func readChunks(globPaths []string) ([]Chunk, error) {
	var chunks []Chunk
	var paths []string

	for _, s := range globPaths {
		match, err := filepath.Glob(s)
		if err != nil {
			return nil, err
		}
		paths = append(paths, match...)
	}

	if len(paths) == 0 {
		return nil, errors.New("no valid dataset specified")
	}

	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}

		index, err := recordio.LoadIndex(f)
		if err != nil {
			return nil, err
		}
		err = f.Close()
		if err != nil {
			return nil, err
		}

		count := index.NumChunks()
		log.Infof("readChunks: file %s has %d chunks", path, count)
		for i := 0; i < count; i++ {
			chunk := Chunk{
				Path:  path,
				Index: *index.ChunkIndex(i),
			}
			chunks = append(chunks, chunk)
		}
	}

	return chunks, nil
}

// SetDataset sets dataset to dispatch for the master server.
//
// SetDataset can be call multiple times. But only the first call will
// be honored.
func (s *Service) SetDataset(globPaths []string, _ *int) error {
	if len(globPaths) == 0 {
		return errors.New("no dataset specified")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.initDone {
		// Already initialized. All trainer will call
		// SetDataset, but we only handle the first one. Treat
		// other calls as successful but do nothing.
		return nil
	}

	chunks, err := readChunks(globPaths)
	if err != nil {
		return err
	}

	s.jobTasks = partition(chunks, s.chunksPerTask)
	s.taskQueues.Todo = s.jobTasks

	err = s.snapshot()
	if err != nil {
		log.Errorln(err)
		return err
	}
	close(s.ready)
	s.initDone = true
	return nil
}

// processFailedTask retry s.failureMax times for failed task.
// return true if all task are done or failed.
func (s *Service) processFailedTask(t taskEntry, epoch int) {
	if t.Task.Meta.Epoch != epoch {
		// new epoch, task launched after the
		// schedule of this timeout check or failed status report.
		return
	}

	defer func() {
		err := s.snapshot()
		if err != nil {
			log.Errorln(err)
		}
	}()

	delete(s.taskQueues.Pending, t.Task.Meta.ID)

	t.NumFailure++
	if t.NumFailure > s.failureMax {
		log.Warningf("Task %v failed %d times, discard.", t.Task, t.NumFailure)
		s.taskQueues.Failed = append(s.taskQueues.Failed, t)
		return
	}

	log.Warningf("Task %v failed %d times, re-dispatch.", t.Task, t.NumFailure)
	s.taskQueues.Todo = append(s.taskQueues.Todo, t)
	return
}

func (s *Service) checkTimeoutFunc(taskID int, epoch int) func() {
	return func() {
		s.mu.Lock()
		defer s.mu.Unlock()

		t, ok := s.taskQueues.Pending[taskID]
		if !ok {
			return
		}

		s.processFailedTask(t, epoch)
	}
}

// must be called with lock held.
func (s *Service) logFields() log.Fields {
	return log.Fields{
		"todoLen":    len(s.taskQueues.Todo),
		"pendingLen": len(s.taskQueues.Pending),
		"doneLen":    len(s.taskQueues.Done),
		"failedLen":  len(s.taskQueues.Failed),
	}
}

// GetTask gets a new task from the service.
// passID is the client side pass count
func (s *Service) GetTask(passID int, task *Task) error {
	select {
	case <-s.ready:
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if passID < s.currPass {
		return ErrPassBefore
	}
	if passID > s.currPass {
		// Client may get run to pass after master when one client faster than the
		// other
		return ErrPassAfter
	}

	if len(s.taskQueues.Todo) == 0 {
		if len(s.taskQueues.Done) == 0 && len(s.taskQueues.Pending) == 0 {
			log.WithFields(s.logFields()).Warningln("All tasks failed, may start next pass")
			return ErrAllTaskFailed
		}
		log.WithFields(s.logFields()).Warningln("No more available task.")
		return ErrNoMoreAvailable
	}

	t := s.taskQueues.Todo[0]
	t.Task.Meta.Epoch++
	s.taskQueues.Todo = s.taskQueues.Todo[1:]
	s.taskQueues.Pending[t.Task.Meta.ID] = t
	err := s.snapshot()
	if err != nil {
		return err
	}

	*task = t.Task
	log.WithFields(s.logFields()).Infof("Task #%v dispatched.", t.Task.Meta)

	time.AfterFunc(s.timeoutDur, s.checkTimeoutFunc(t.Task.Meta.ID, t.Task.Meta.Epoch))
	return nil
}

// TaskFinished tell the service that a task is finished.
func (s *Service) TaskFinished(taskID int, dummy *int) error {
	select {
	case <-s.ready:
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	t, ok := s.taskQueues.Pending[taskID]
	if !ok {
		log.WithFields(s.logFields()).Warningln("Pending task #%d not found.", taskID)
		return nil
	}

	// task finished, reset timeout
	t.NumFailure = 0
	s.taskQueues.Done = append(s.taskQueues.Done, t)
	delete(s.taskQueues.Pending, taskID)

	log.WithFields(s.logFields()).Infof("Task #%d finished.", taskID)
	if len(s.taskQueues.Todo) == 0 && len(s.taskQueues.Pending) == 0 {
		// increase master side pass count if all tasks finished
		s.currPass++
		s.taskQueues.Todo = s.jobTasks
		s.taskQueues.Done = []taskEntry{}
		// TODO(typhoonzero): deal with failed tasks
		s.taskQueues.Failed = []taskEntry{}
		log.WithFields(s.logFields()).Warningf("all task finished, add new pass data, newpass: %d.", s.currPass)
	}

	err := s.snapshot()
	if err != nil {
		log.Errorln(err)
	}
	return err
}

// TaskFailed tells the service that a task is failed.
func (s *Service) TaskFailed(meta TaskMeta, dummy *int) error {
	select {
	case <-s.ready:
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	t, ok := s.taskQueues.Pending[meta.ID]
	if !ok {
		log.WithFields(s.logFields()).Warningln("TaskFailed:Pending task #%v not found.", t.Task.Meta)
		return nil
	}

	s.processFailedTask(t, meta.Epoch)
	return nil
}

// SaveModelRequest is the request for saving model
type SaveModelRequest struct {
	TrainerID string
	BlockDur  time.Duration
}

// RequestSaveModel requests the master server to approve the caller
// to save the model.
func (s *Service) RequestSaveModel(req SaveModelRequest, need *bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if req.TrainerID == "" {
		return errors.New("trainer id is empty")
	}

	if s.savingTrainer == "" {
		*need = true
	} else {
		if req.TrainerID == s.savingTrainer {
			// save trainer asked to save model again
			*need = true
		} else {
			*need = false
		}
	}

	if *need {
		s.savingTrainer = req.TrainerID
		time.AfterFunc(req.BlockDur, func() {
			s.mu.Lock()
			s.savingTrainer = ""
			s.mu.Unlock()
		})
	}

	return nil
}
