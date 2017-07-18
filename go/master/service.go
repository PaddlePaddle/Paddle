package master

import (
	"bytes"
	"compress/gzip"
	"encoding/gob"
	"errors"
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
	ready         chan struct{}
	store         Store

	mu         sync.Mutex
	initDone   bool
	taskQueues taskQueues

	numPasses int
	currPass  int
}

func partition(chunks []Chunk, chunksPerTask int) []taskEntry {
	id := 0
	if chunksPerTask <= 0 {
		chunksPerTask = 1
	}

	var result []taskEntry
	var cur taskEntry
	for i, c := range chunks {
		if i%chunksPerTask == 0 && len(cur.Task.Chunks) > 0 {
			cur.Task.Meta.ID = id
			id++
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
	s.numPasses = 0
	s.currPass = 0
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
	// TOOD(helin): etcd request has a size limit, so the snapshot
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

// SetDatasetRequest is a request for setting datasets and numpaasses
type SetDatasetRequest struct {
	GlobPaths []string
	NumPasses int
}

// SetDataset sets dataset to dispatch for the master server.
//
// SetDataset can be call multiple times. But only the first call will
// be honored.
func (s *Service) SetDataset(request SetDatasetRequest, dummy *int) error {
	globPaths := request.GlobPaths
	s.numPasses = request.NumPasses
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

	s.taskQueues.Todo = partition(chunks, s.chunksPerTask)

	err = s.snapshot()
	if err != nil {
		log.Errorln(err)
		return err
	}

	close(s.ready)
	s.initDone = true
	return nil
}

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

	log.Warningf("Task %v failed %d times, discard.", t.Task, t.NumFailure)
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

// updateQueuePasses check if tasks are done, move to todo to add another pass
// IMPORTANT: must run under lock
func (s *Service) updateQueuePasses() error {
	if len(s.taskQueues.Todo) == 0 && len(s.taskQueues.Pending) == 0 && len(s.taskQueues.Done) > 0 {
		if s.currPass >= s.numPasses-1 {
			// FIXME: stop task dispatching by return error
			return errors.New("all task done")
		}
		log.WithFields(s.logFields()).Infof("adding new pass %d", s.currPass)
		s.taskQueues.Todo = s.taskQueues.Done
		s.taskQueues.Done = nil
		if len(s.taskQueues.Failed) > 0 {
			s.taskQueues.Todo = append(s.taskQueues.Todo, s.taskQueues.Failed...)
		}
		s.currPass++
	}
	return errors.New("no more available task")
}

// GetTask gets a new task from the service.
func (s *Service) GetTask(dummy int, task *Task) error {
	select {
	case <-s.ready:
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.taskQueues.Todo) == 0 {
		if len(s.taskQueues.Done) == 0 {
			if len(s.taskQueues.Pending) == 0 {
				err := errors.New("all task failed")
				log.WithFields(s.logFields()).Warningln("All tasks failed.")
				return err
			}

			// TODO(helin): client need to retry in this
			// error case. Gotcha: RPC client can't
			// compare returned error with predefined
			// errors like io.EOF, because the error
			// instance deserialized from RPC is a
			// different instance than the error defined
			// in package. So we need to figure out a way
			// for client to check this error correctly.
			err := errors.New("no more available task")
			log.WithFields(s.logFields()).Warningln("No more available task.")
			return err
		}
		return s.updateQueuePasses()
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
	// update queues if pass finishes
	errPass := s.updateQueuePasses()

	err := s.snapshot()
	if err != nil {
		log.Errorln(err)
	}
	// return updateQueuePasses errors
	if errPass.Error() == "no more available task" {
		return nil
	}
	return errPass
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
	// update queues if pass finishes
	errPass := s.updateQueuePasses()
	if errPass.Error() == "no more available task" {
		return nil
	}
	return errPass
}
