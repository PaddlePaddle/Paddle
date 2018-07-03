// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
	"context"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	log "github.com/inconshreveable/log15"
)

const (
	// DefaultLockPath is the default etcd master lock path.
	DefaultLockPath = "/master/lock"
	// DefaultStatePath is the default etcd key for master state.
	DefaultStatePath = "/master/state"
	// DefaultAddrPath is the default etcd key for master address.
	DefaultAddrPath = "/master/addr"
)

// EtcdClient is the etcd client that the master uses for fault
// tolerance and service registry.
type EtcdClient struct {
	lockPath  string
	statePath string
	client    *clientv3.Client
	lock      *concurrency.Mutex
	sess      *concurrency.Session
}

// NewEtcdClient creates a new EtcdClient.
func NewEtcdClient(endpoints []string, addr string, lockPath, addrPath, statePath string, ttlSec int) (*EtcdClient, error) {
	log.Debug("Connecting to etcd", log.Ctx{"endpoint": endpoints})
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		return nil, err
	}

	sess, err := concurrency.NewSession(cli, concurrency.WithTTL(ttlSec))
	if err != nil {
		return nil, err
	}

	lock := concurrency.NewMutex(sess, lockPath)
	// It's fine for the lock to get stuck, in this case we have
	// multiple master servers running (only configured to have
	// one master running, but split-brain problem may cause
	// multiple master servers running), and the cluster management
	// software will kill one of them.
	log.Info("Trying to acquire lock.", log.Ctx{"path": lockPath})
	err = lock.Lock(context.TODO())
	if err != nil {
		return nil, err
	}
	log.Info("Successfully acquired lock at %s.", log.Ctx{"path": lockPath})

	put := clientv3.OpPut(addrPath, addr)
	resp, err := cli.Txn(context.Background()).If(lock.IsOwner()).Then(put).Commit()
	if err != nil {
		return nil, err
	}

	if !resp.Succeeded {
		log.Crit("No longer owns the master lock. Exiting.")
		panic("No longer owns the master lock. Exiting.")
	}

	e := &EtcdClient{
		lockPath:  lockPath,
		statePath: statePath,
		client:    cli,
		lock:      lock,
		sess:      sess,
	}

	return e, nil
}

// Save saves the state into the etcd.
func (e *EtcdClient) Save(state []byte) error {
	ctx := context.TODO()
	put := clientv3.OpPut(e.statePath, string(state))
	resp, err := e.client.Txn(ctx).If(e.lock.IsOwner()).Then(put).Commit()
	if err != nil {
		return err
	}

	if !resp.Succeeded {
		log.Error("No longer owns the lock, trying to lock again")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		err := e.lock.Lock(ctx)
		cancel()
		if err != nil {
			// We lost the master lock and can not acquire
			// it back, it means some other master is
			// already started. We don't want cluster
			// management system to kill the master server
			// who is holding the lock and running
			// correctly. So the most feasible solution is
			// to kill current master server. The current
			// state is not saved, but the trainer's RPC
			// call will fail, so the trainer will retry.
			log.Crit("Could not acquire the lock at %s: %v. Exiting.", log.Ctx{"path": e.lockPath, "error": err})
			panic("Could not acquire the lock at %s: %v. Exiting.")
		}
		log.Info("Successfully acquired lock at %s.", e.lockPath)
		return e.Save(state)
	}

	return nil
}

// Load loads the state from etcd.
func (e *EtcdClient) Load() ([]byte, error) {
	ctx := context.TODO()
	get := clientv3.OpGet(e.statePath)

	resp, err := e.client.Txn(ctx).If(e.lock.IsOwner()).Then(get).Commit()
	if err != nil {
		return nil, err
	}

	if !resp.Succeeded {
		log.Error("No longer owns the lock, trying to lock and load again.")
		err = e.lock.Lock(context.Background())
		if err != nil {
			return nil, err
		}

		return e.Load()
	}

	kvs := resp.Responses[0].GetResponseRange().Kvs
	if len(kvs) == 0 {
		// No state exists
		return nil, nil
	}

	state := kvs[0].Value
	return state, nil
}

// Shutdown shuts down the etcd client gracefully.
func (e *EtcdClient) Shutdown() error {
	err := e.sess.Close()
	newErr := e.client.Close()
	if newErr != nil {
		if err == nil {
			err = newErr
		} else {
			log.Error("shutdown error", log.Ctx{"error": newErr})
		}
	}

	return err
}

// GetKey gets the value by the specify key.
func GetKey(c *clientv3.Client, key string, timeout time.Duration) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	resp, err := c.Get(ctx, key)
	cancel()
	if err != nil {
		return "", err
	}
	kvs := resp.Kvs
	if len(kvs) == 0 {
		return "", nil
	}
	v := kvs[0].Value
	return string(v), nil
}

// watchKey watches the specify key and send to valChan if there is some event.
func watchKey(c *clientv3.Client, key string, valChan chan<- string) {
	rch := c.Watch(context.Background(), key)
	for wresp := range rch {
		for _, ev := range wresp.Events {
			// if received event is DELETE, the value will be an empty string
			log.Info("received event.", log.Ctx{"type": ev.Type, "key": ev.Kv.Key, "value": ev.Kv.Value})
			valChan <- string(ev.Kv.Value)
		}
	}
}
