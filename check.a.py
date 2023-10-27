import numpy as np
p = np.load('sdpa.npy')
t = np.load('t_sdpa.npy')
#t = t.transpose(1,0,2)
#t = t.reshape(t.shape[0], -1)
diff = np.max(np.abs(p - t))
print(diff)
import pdb; pdb.set_trace()
