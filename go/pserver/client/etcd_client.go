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

package client

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/PaddlePaddle/Paddle/go/pserver"
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	log "github.com/sirupsen/logrus"
)

const (
	defaultEtcdTimeout time.Duration = 5 * time.Second

	initLockPath = "/init_ps/lock"
	initDonePath = "/init_ps/done"
	initDoneVal  = "1"
)

// Etcd is used by pserver client that is a part of trainer process.
// TODO:
// 1. add watcher to watch the change state of pservers.
type Etcd struct {
	client    *clientv3.Client
	timeout   time.Duration
	endpoints []string
	lock      *concurrency.Mutex
}

// Desired read ps desired number from etcd.
func (e *Etcd) Desired() int {
	var psDesired int
	for {
		ctx, cancel := context.WithTimeout(context.Background(), e.timeout)
		resp, err := e.client.Get(ctx, pserver.PsDesired)
		cancel()
		if err != nil {
			log.Errorf("Get ps dresire number failed! recnnectiong..., %v", err)
			time.Sleep(e.timeout)
			continue
		}

		kvs := resp.Kvs
		if len(kvs) == 0 {
			log.Infoln("Waiting for ps desired registered ...")
			time.Sleep(e.timeout)
			continue
		}

		psDesired, err = strconv.Atoi(string(resp.Kvs[0].Value))
		if err != nil {
			log.Errorf("psDesired %d invalid %v", psDesired, err)
			time.Sleep(e.timeout)
			continue
		}

		log.Debugf("Get psDesired number: %d", psDesired)
		break
	}
	return psDesired
}

// List return the pserver list read from etcd.
func (e *Etcd) List() []Server {
	psDesired := e.Desired()

	servers := make([]Server, psDesired)
	for {
		for i := 0; i < psDesired; i++ {
			ctx, cancel := context.WithTimeout(context.Background(), e.timeout)
			psKey := pserver.PsPath + strconv.Itoa(i)
			log.Debugf("checking %s", psKey)
			resp, err := e.client.Get(ctx, psKey)
			cancel()
			if err != nil {
				log.Infof("Get psKey= %s error, %v", psKey, err)
				time.Sleep(e.timeout)
				continue
			}
			kvs := resp.Kvs
			if len(kvs) == 0 {
				log.Infof("Waiting for ps addr registered ...")
				time.Sleep(e.timeout)
				continue
			}

			psAddr := string(resp.Kvs[0].Value)
			// TODO(Longfei) check the ps address
			if psAddr == "" {
				log.Infof("Get psKey = %s, psAddr is empty", psKey)
				time.Sleep(e.timeout)
				continue
			}
			log.Debugf("got value (%s) for key: %s", psAddr, psKey)
			servers[i].Index = i
			servers[i].Addr = psAddr
		}
		break
	}
	return servers
}

// NewEtcd create a etcd client to return the state of pserver on etcd.
func NewEtcd(endpoints string) *Etcd {
	ep := strings.Split(endpoints, ",")
	var cli *clientv3.Client
	var err error
	for {
		cli, err = clientv3.New(clientv3.Config{
			Endpoints:   ep,
			DialTimeout: defaultEtcdTimeout,
		})
		if err != nil {
			log.Errorf("Init etcd connection failed: %v", err)
			time.Sleep(defaultEtcdTimeout)
			continue
		}
		break
	}
	log.Infof("Connected to etcd: %s\n", endpoints)
	client := &Etcd{
		client:    cli,
		timeout:   defaultEtcdTimeout,
		endpoints: ep,
	}
	return client
}

// Select indicates if the current trainer is selected to initialize
// the pserver parameters.
func (e *Etcd) Select() (bool, error) {
	sess, err := concurrency.NewSession(e.client, concurrency.WithTTL(5))
	if err != nil {
		return false, err
	}

	lock := concurrency.NewMutex(sess, initLockPath)
	log.Infof("Trying to acquire lock at %s.", initLockPath)
	// Do not use timeout context here, since we don't know how
	// long does it take for other trainers to initialize the
	// parameters.
	err = lock.Lock(context.Background())
	if err != nil {
		return false, err
	}
	log.Infof("Successfully acquired lock at %s.", initLockPath)

	get := clientv3.OpGet(initDonePath)
	ctx, cancel := context.WithTimeout(context.Background(), e.timeout)
	tresp, err := e.client.Txn(ctx).If(lock.IsOwner()).Then(get).Commit()
	cancel()
	if err != nil {
		return false, err
	}

	if !tresp.Succeeded {
		return false, errors.New("no longer the owner of the lock")
	}

	resp := tresp.Responses[0].GetResponseRange()

	if len(resp.Kvs) == 0 {
		// Key value not set, select current trainer.
		e.lock = lock
		log.Infoln("Trainer selected.")
		return true, nil
	}

	if string(resp.Kvs[0].Value) == initDoneVal {
		log.Infoln("Initialization is already done.")
		ctx, cancel = context.WithTimeout(context.Background(), e.timeout)
		err = lock.Unlock(ctx)
		cancel()
		if err != nil {
			log.Errorln(err)
		}
		return false, nil
	}

	return false, fmt.Errorf("key %s have unexpected value: %v", initDonePath, resp.Kvs[0].Value)
}

// Done indicates the parameter initialization process is done.
func (e *Etcd) Done() error {
	if e.lock == nil {
		return errors.New("lock is nil, Done called unexpectedly")
	}

	put := clientv3.OpPut(initDonePath, initDoneVal)
	ctx, cancel := context.WithTimeout(context.Background(), e.timeout)
	tresp, err := e.client.Txn(ctx).If(e.lock.IsOwner()).Then(put).Commit()
	cancel()
	if err != nil {
		return err
	}

	if !tresp.Succeeded {
		return errors.New("no longer the owner of the lock")
	}

	ctx, cancel = context.WithTimeout(context.Background(), e.timeout)
	err = e.lock.Unlock(ctx)
	cancel()
	if err != nil {
		log.Errorln(err)
	} else {
		e.lock = nil
	}

	return nil
}

// Close closes the etcd client.
func (e *Etcd) Close() error {
	var err error
	if e.lock != nil {
		ctx, cancel := context.WithTimeout(context.Background(), e.timeout)
		err = e.lock.Unlock(ctx)
		cancel()
		if err == nil {
			e.lock = nil
		}
	}

	cErr := e.client.Close()
	if cErr != nil {
		if err != nil {
			log.Errorln(cErr)
			return err
		}
		return cErr
	}

	return err
}
