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

package pserver

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/PaddlePaddle/Paddle/go/utils/networkhelper"
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	log "github.com/inconshreveable/log15"
)

const (
	// PsDesired is etcd path for store desired pserver count
	PsDesired = "/ps_desired"
	// PsPath is the base dir for pserver to store their addr
	PsPath = "/ps/"
	// PsCheckpoint is the etcd path for store checkpoints information
	PsCheckpoint = "/checkpoints/"

	retryTimeout = 5 * time.Second
)

// EtcdClient is the etcd client that the pserver uses for fault
// tolerance, service registry and coordination.
type EtcdClient struct {
	numPservers int
	endpoints   string
	client      *clientv3.Client
	sess        *concurrency.Session
	dialTimeout time.Duration
	ttlSec      int
	// FIXME: ensure GetExternalIP gets the correct ip for trainers to connect.
	externalIP string
	// desired number of pservers in the job.
	// assume desired will not change during one training job.
	desired int
}

// NewEtcdClient creates an EtcdClient
func NewEtcdClient(endpoints string, numPservers int, dialtimeout time.Duration, ttlSec int) *EtcdClient {
	return &EtcdClient{
		dialTimeout: dialtimeout,
		ttlSec:      ttlSec,
		numPservers: numPservers,
		endpoints:   endpoints,
	}
}

// Register registers the pserver on etcd
//
// Register returns the index of the current pserver.
func (e *EtcdClient) Register(port int) (int, error) {
	var err error
	e.externalIP, err = networkhelper.GetExternalIP()
	if err != nil {
		return 0, err
	}

	// initialize connection to etcd.
	ep := strings.Split(e.endpoints, ",")
	for {
		cli, err := clientv3.New(clientv3.Config{
			Endpoints:   ep,
			DialTimeout: e.dialTimeout,
		})
		if err != nil {
			log.Error("connect to etcd error", log.Ctx{"error": err})
			time.Sleep(retryTimeout)
			continue
		}
		e.client = cli
		sess, err := concurrency.NewSession(cli, concurrency.WithTTL(e.ttlSec))
		if err != nil {
			log.Error("create etcd session error", log.Ctx{"error": err})
			time.Sleep(retryTimeout)
			continue
		}
		e.sess = sess
		log.Debug("connected to etcd", log.Ctx{"endpoint": e.endpoints})
		break
	}
	// init /ps_desired using transaction, for multiple pservers may want to write
	// it at the same time.
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_, err := e.initDesiredPservers(ctx, e.numPservers)
		cancel()
		if err != nil {
			log.Warn("pserver init error", log.Ctx{"error": err, "num pservers": e.numPservers})
			time.Sleep(retryTimeout)
			continue
		}
		break
	}
	// TODO: when implementing extending or reducing pservers, /ps_desired is
	// changed, then we need to watch /ps_desired node for events. For now, just
	// write once when init and read from it.
	// wait and set s.desired init value
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		resp, err := e.client.Get(ctx, PsDesired)
		cancel()
		if err != nil {
			log.Error("get etcd key error", log.Ctx{"key": PsDesired, "error": err})
			time.Sleep(retryTimeout)
			continue
		}
		if len(resp.Kvs) != 0 {
			e.desired, err = strconv.Atoi(string(resp.Kvs[0].Value))
			if err != nil {
				log.Error(
					"psDesired atoi error",
					log.Ctx{"error": err, "value": string(resp.Kvs[0].Value)},
				)
				time.Sleep(retryTimeout)
				// NOTE: wait util ps_desired value change
				continue
			}
			break
		}
	}

	var pserverIdx int
	// try register pserver node on etcd
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		var err error
		pserverIdx, err = e.registerPserverEtcd(ctx, port)
		cancel()
		if err != nil {
			log.Warn("register pserver on etcd error", log.Ctx{"error": err})
			time.Sleep(retryTimeout)
			continue
		}
		break
	}

	return pserverIdx, nil
}

func (e *EtcdClient) initDesiredPservers(ctx context.Context, numPservers int) (*clientv3.TxnResponse, error) {
	return concurrency.NewSTM(e.client, func(c concurrency.STM) error {
		dsStr := c.Get(PsDesired)
		if dsStr == "" {
			c.Put(PsDesired, strconv.Itoa(numPservers), clientv3.WithLease(e.sess.Lease()))
		}
		return nil
	}, concurrency.WithAbortContext(ctx), concurrency.WithIsolation(concurrency.RepeatableReads))
}

// registerPserverEtcd registers pserver node on etcd using transaction.
func (e *EtcdClient) registerPserverEtcd(ctx context.Context, port int) (int, error) {
	var idx int
	_, err := concurrency.NewSTM(e.client, func(c concurrency.STM) error {
		registered := false
		for i := 0; i < e.desired; i++ {
			psKey := PsPath + strconv.Itoa(i)
			ps := c.Get(psKey)
			log.Debug(
				"register pserver got value",
				log.Ctx{"value": ps, "key": psKey},
			)

			if ps == "" {
				// find the first id and write info
				pserverAddr := e.externalIP + ":" + strconv.Itoa(port)
				c.Put(psKey, pserverAddr, clientv3.WithLease(e.sess.Lease()))
				log.Debug("register finished", log.Ctx{"key": psKey, "value": pserverAddr})
				idx = i
				registered = true
				break
			}
		}
		if registered {
			return nil
		}
		return errors.New("not registered, may due to already have enough pservers")
	}, concurrency.WithAbortContext(ctx), concurrency.WithIsolation(concurrency.RepeatableReads))

	if err != nil {
		return 0, err
	}

	return idx, nil
}

// GetKey gets the value by the specified key
func (e *EtcdClient) GetKey(key string, timeout time.Duration) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	resp, err := e.client.Get(ctx, key)
	cancel()
	if err != nil {
		return []byte{}, err
	}

	kvs := resp.Kvs
	if len(kvs) == 0 {
		return []byte{}, nil
	}
	v := kvs[0].Value
	return v, nil
}

// PutKey put into etcd with value by key specified
func (e *EtcdClient) PutKey(key string, value []byte, timeout time.Duration, withLease bool) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	var err error
	if withLease {
		_, err = e.client.Put(ctx, key, string(value), clientv3.WithLease(e.sess.Lease()))
	} else {
		_, err = e.client.Put(ctx, key, string(value))
	}
	cancel()
	return err
}

// Shutdown shuts down the etcd client gracefully.
func (e *EtcdClient) Shutdown() error {
	var err error
	if e.sess != nil {
		err = e.sess.Close()
	}

	if e.client != nil {
		newErr := e.client.Close()
		if newErr != nil {
			if err != nil {
				log.Error("shutdown error", log.Ctx{"error": newErr})
			} else {
				err = newErr
			}
		}
	}
	return err
}
