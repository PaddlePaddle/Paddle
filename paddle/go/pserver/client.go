package pserver

// ElementType is the type of elements of a Parameter.
type ElementType int

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

// Client is the client to parameter servers.
type Client struct {
}

// NewClient creates a new client.
func NewClient(addr string) *Client {
	return &Client{}
}

// BeginInitParams begins to initialize parameters on parameter
// servers.
//
// BeginInitParams will be called from multiple trainers, only one
// trainer will be selected to initialize the parameters on parameter
// servers. Other trainers will be blocked until the initialization is
// done, and they need to get the initialized parameters from
// parameter servers using GetParams.
func (c *Client) BeginInitParams(pserverConfigProto []byte) (selected bool, err error) {
	return true, nil
}

// InitParam initializes the parameter on parameter servers.
func (c *Client) InitParam(paramWithConfigs ParameterWithConfig) error {
	return nil
}

// FinishInitParams tells parameter servers client has sent all
// parameters to parameter servers as initialization.
func (c *Client) FinishInitParams() error {
	return nil
}

// SendGrads sends gradients to parameter servers for updating
// parameters.
func (c *Client) SendGrads(grads []Gradient) error {
	return nil
}

// GetParams gets parameters from parameter servers.
func (c *Client) GetParams(names []string) ([]Parameter, error) {
	return nil, nil
}

// SaveModel indicates parameters to save the parameter to the given
// path.
func (c *Client) SaveModel(path string) error {
	return nil
}

// Cleanup cleans up the client states.
func (c *Client) Cleanup() {
}
