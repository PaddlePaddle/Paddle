package master

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/PaddlePaddle/Paddle/go/connection"
	"github.com/PaddlePaddle/recordio"
	"github.com/coreos/etcd/clientv3"
	log "github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

const masterAddrPath = "/master"

// Client is the client of the master server.
type Client struct {
	conn *connection.Conn
	ch   chan []byte
}

// EtcdClient is the client of
type EtcdClient struct {
	client    *clientv3.Client
	endpoints []string
	ch        chan string
	timeout   time.Duration
}

// NewClient creates a new Client.
//
// bufSize is the record buffer size. NextRecord will read from this
// buffer.
func NewClient(addrCh <-chan string, bufSize int) *Client {
	c := &Client{}
	c.conn = connection.New()
	c.ch = make(chan []byte, bufSize)
	go c.monitorMaster(addrCh)
	go c.getRecords()
	return c
}

func (c *Client) getRecords() {
	for {
		t, err := c.getTask()
		if err != nil {
			// TODO(helin): wait before move on with next
			// getTask call.
			log.Errorln(err)
			continue
		}

		for _, chunk := range t.Chunks {
			f, err := os.Open(chunk.Path)
			if err != nil {
				log.Errorln(err)
				continue
			}

			s := recordio.NewRangeScanner(f, &chunk.Index, -1, -1)
			for s.Scan() {
				c.ch <- s.Record()
			}

			if s.Err() != nil {
				log.Errorln(err, chunk.Path)
			}

			err = f.Close()
			if err != nil {
				log.Errorln(err)
			}
		}

		// We treat a task as finished whenever the last data
		// instance of the task is read. This is not exactly
		// correct, but a reasonable approximation.
		c.taskFinished(t.ID)
	}
}

func (c *Client) monitorMaster(addrCh <-chan string) {
	lastMaster := ""
	for curMaster := range addrCh {
		// connect to the new address once address changed.
		if curMaster != lastMaster {
			if curMaster == "" {
				err := c.conn.Close()
				fmt.Printf("close conn error: %s", err)
				if err != nil {
					log.Errorln(err)
				}
			} else {
				err := c.conn.Connect(curMaster)
				if err != nil {
					log.Errorln(err)

					// connect to addr failed, set
					// to last known addr in order
					// to retry next time.
					curMaster = lastMaster
				}
			}
		}
		lastMaster = curMaster
	}
}

// SetDataset set dataset for the master server to dispatch.
//
// SetDataset can be call multiple times from different nodes. But
// only the first call will be honored.
func (c *Client) SetDataset(globPaths []string) error {
	return c.conn.Call("Service.SetDataset", globPaths, nil)
}

// getTask gets a new task from the master server.
func (c *Client) getTask() (Task, error) {
	var t Task
	err := c.conn.Call("Service.GetTask", 0, &t)
	return t, err
}

// TaskFinished tells the master server a task is finished.
func (c *Client) taskFinished(taskID int) error {
	return c.conn.Call("Service.TaskFinished", taskID, nil)
}

// NextRecord returns next record in the dataset.
//
// NextRecord will block until the next record is available. It is
// thread-safe.
func (c *Client) NextRecord() []byte {
	return <-c.ch
}

// NewEtcdClient create a new master client by etcd
//
// endpoints is the endpoints for etcd and separated by ",", such as
// "172.0.1.0:2379,172.0.1.1:2379"
// timeout is the timeout for etcd calls
// bufSize is the record buffer size. NextRecord will read from this buffer.
func NewEtcdClient(endpoints string, timeout int, bufSize int) *Client {
	t := time.Second * time.Duration(timeout)
	ep := strings.Split(endpoints, ",")
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   ep,
		DialTimeout: t,
	})
	if err != nil {
		log.Errorf("Init etcd connection failed: %v", err)
		panic(err)
	}
	log.Debugf("Connected to etcd: %s\n", endpoints)
	etcdClient := EtcdClient{
		client:    cli,
		timeout:   t,
		endpoints: ep,
	}
	etcdClient.ch = make(chan string)
	c := NewClient(etcdClient.ch, bufSize)
	//go etcdClient.monitorMasterAddr()
	etcdClient.initMasterAddr(masterAddrPath)
	go etcdClient.monitorMasterAddr()
	return c
}
func (e *EtcdClient) initMasterAddr(key string) {
	for {
		ctx, cancel := context.WithTimeout(context.Background(), e.timeout)
		resp, err := e.client.Get(ctx, masterAddrPath)
		cancel()
		if err != nil {
			log.Errorf("etcd get key: %s failed: %s, sleep for %d seconds and reconnect...",
				key, err, e.timeout)
			time.Sleep(e.timeout)
			err = e.client.Close()
			if err != nil {
				log.Error(err)
			}
			e.client, err = clientv3.New(clientv3.Config{
				Endpoints:   e.endpoints,
				DialTimeout: e.timeout,
			})
			if err != nil {
				log.Error(err)
			}
			continue
		}
		if len(resp.Kvs) == 0 {
			log.Errorf("etcd key: %s does not exists, sleep %d seconds...", key, e.timeout/time.Second)
			time.Sleep(e.timeout)
			continue
		}
		mAddr := string(resp.Kvs[0].Value)
		e.ch <- mAddr
		break
	}
	fmt.Println("init master addr finished.")
}
func (e *EtcdClient) monitorMasterAddr() {
	rch := e.client.Watch(context.Background(), masterAddrPath)
	for wresp := range rch {
		for _, ev := range wresp.Events {
			// if event type is DELETE, ev.Kv.Value will be a empty string and Client
			// will close the connection
			e.ch <- string(ev.Kv.Value)
		}
	}
}
