package master

import (
	"os"

	"github.com/PaddlePaddle/Paddle/go/connection"
	"github.com/PaddlePaddle/recordio"
	log "github.com/sirupsen/logrus"
)

// Client is the client of the master server.
type Client struct {
	conn *connection.Conn
	ch   chan []byte
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
