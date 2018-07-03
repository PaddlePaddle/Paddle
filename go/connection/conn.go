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

package connection

import (
	"errors"
	"net/rpc"
	"sync"

	log "github.com/sirupsen/logrus"
)

// TODO(helin): add TCP re-connect logic

// Conn is a connection to a parameter server
type Conn struct {
	mu       sync.Mutex
	client   *rpc.Client
	waitConn chan struct{}
}

// New creates a new connection.
func New() *Conn {
	c := &Conn{}
	return c
}

// Close closes the connection.
func (c *Conn) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.client == nil {
		return nil
	}

	return c.client.Close()
}

// Connect connects the connection to a address.
func (c *Conn) Connect(addr string) error {
	c.mu.Lock()
	if c.client != nil {
		err := c.client.Close()
		if err != nil {
			c.mu.Unlock()
			return err
		}

		c.client = nil
	}
	c.mu.Unlock()

	client, err := rpc.DialHTTP("tcp", addr)
	if err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.client == nil {
		c.client = client
		if c.waitConn != nil {
			close(c.waitConn)
			c.waitConn = nil
		}
	} else {
		err := client.Close()
		if err != nil {
			log.Errorln(err)
		}

		return errors.New("client already set from a concurrent goroutine")
	}

	return nil
}

// TODO(helin): refactor Call to be able to perform given retry
// policy.

// Call make a RPC call.
//
// Call will be blocked until the connection to remote RPC service
// being established.
func (c *Conn) Call(serviceMethod string, args interface{}, reply interface{}) error {
	c.mu.Lock()
	client := c.client
	var waitCh chan struct{}
	if client == nil {
		if c.waitConn != nil {
			waitCh = c.waitConn
		} else {
			waitCh = make(chan struct{})
			c.waitConn = waitCh
		}
	}
	c.mu.Unlock()

	if waitCh != nil {
		// wait until new connection being established
		<-waitCh
		return c.Call(serviceMethod, args, reply)
	}

	return client.Call(serviceMethod, args, reply)
}
