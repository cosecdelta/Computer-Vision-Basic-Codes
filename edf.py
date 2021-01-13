### EDF --- An Autograd Engine for instruction
## (based on joint discussions with David McAllester)

import numpy as np
from scipy.signal import convolve2d as conv

# Global list of different kinds of components
ops = []
params = []
values = []


# Global forward
def Forward():
    for c in ops: c.forward()

# Global backward    
def Backward(loss):
    for c in ops:
        c.grad = np.zeros_like(c.top)
    for c in params:
        c.grad = np.zeros_like(c.top)

    loss.grad = np.ones_like(loss.top)
    for c in ops[::-1]: c.backward() 

# SGD
def SGD(lr):
    for p in params:
        p.top = p.top - lr*p.grad
    

## Fill this out        
def init_momentum():
    for p in params:
        p.prev_grad = np.zeros_like(p.top)


# ## Fill this out

def momentum(lr, mom=0.9):
    for p in params:
        p.grad = p.grad + mom*p.prev_grad
        p.prev_grad = p.grad
        p.top = p.top - lr*p.grad
        
    
###################### Different kinds of nodes

# Values (Inputs)
class Value:
    def __init__(self):
        values.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()

# Parameters (Weights we want to learn)
class Param:
    def __init__(self):
        params.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()


### Operations

# Add layer (x + y) where y is same shape as x or is 1-D
class add:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = self.x.top + self.y.top

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad

        if self.y in ops or self.y in params:
            if len(self.y.top.shape) < len(self.grad.shape):
                ygrad = np.sum(self.grad,axis=tuple(range(len(self.grad.shape)-1)))
            else:
                ygrad= self.grad
            self.y.grad = self.y.grad + ygrad

# Matrix multiply (fully-connected layer)
class matmul:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = np.matmul(self.x.top,self.y.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.matmul(self.y.top,self.grad.T).T
        if self.y in ops or self.y in params:
            self.y.grad = self.y.grad + np.matmul(self.x.top.T,self.grad)


# Rectified Linear Unit Activation            
class RELU:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.maximum(self.x.top,0)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad * (self.top > 0)


# Reduce to mean
class mean:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.mean(self.x.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.top) / np.float32(np.prod(self.x.top.shape))



# Soft-max + Loss (per-row / training example)
class smaxloss:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        y = self.x.top
        y = y - np.amax(y,axis=1,keepdims=True)
        yE = np.exp(y)
        yS = np.sum(yE,axis=1,keepdims=True)
        y = y - np.log(yS); yE = yE / yS

        truey = np.int64(self.y.top)
        self.top = -y[range(len(truey)),truey]
        self.save = yE

    def backward(self):
        if self.x in ops or self.x in params:
            truey = np.int64(self.y.top)
            self.save[range(len(truey)),truey] = self.save[range(len(truey)),truey] - 1.
            self.x.grad = self.x.grad + np.expand_dims(self.grad,-1)*self.save
        # No backprop to labels!    

# Compute accuracy (for display, not differentiable)        
class accuracy:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        truey = np.int64(self.y.top)
        self.top = np.float32(np.argmax(self.x.top,axis=1)==truey)

    def backward(self):
        pass


# Downsample by 2    
class down2:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = self.x.top[:,::2,::2,:]

    def backward(self):
        if self.x in ops or self.x in params:
            grd = np.zeros_like(self.x.top)
            grd[:,::2,::2,:] = self.grad
            self.x.grad = self.x.grad + grd


# Flatten (conv to fc)
class flatten:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = np.reshape(self.x.top,[self.x.top.shape[0],-1])

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.top.shape)
            
# Convolution Layer
## Fill this out
class conv2:

    def __init__(self,x,k):
        ops.append(self)
        self.x = x
        self.k = k

    def forward(self):
        k_size = self.k.top.shape[0]
        img = self.x.top
        K = self.k.top
        # if (img.shape[1] - k_size)%2 != 0:
        #     img = np.pad(img,((0,0),(0,1),(1,0),(0,0)), mode='constant')
        
        #print(img.shape)
        img_conv = np.zeros((img.shape[0],img.shape[1] - k_size + 1, img.shape[2] - k_size + 1,K.shape[3]))
        temp_conv = np.zeros((img.shape[0], img.shape[1] - k_size + 1, img.shape[2] - k_size + 1, K.shape[2]))
        
        w = img_conv.shape[1]
        h = img_conv.shape[2]
        
        for c in range(K.shape[3]):
            temp_conv = np.zeros((img.shape[0], img.shape[1] - k_size + 1, img.shape[2] - k_size + 1, K.shape[2]))
            for i in range(k_size):
                for j in range(k_size):
                    temp_conv = temp_conv + img[:,i:i+w, j:j+h,:].copy() * K[(k_size-1)-i, (k_size-1)-j,:,c]
            
            temp_conv = np.sum(temp_conv, axis = 3)
            img_conv[:,:,:,c] = temp_conv
                    
        
        self.top = img_conv
        

    def backward(self):
        if self.x in ops or self.x in params:
            img = self.x.top
            dy = self.grad
            weight_new = self.k.top
            # weight = np.fliplr(weight)
            # weight = np.flipud(weight)
            weight = np.rot90(weight_new, 2 ,(0,1))
            k_size = weight.shape[0]
            npad = ((0, 0), (k_size-1, k_size-1), (k_size-1, k_size-1),(0,0))
            dy_res = np.pad(dy, pad_width = npad, mode='constant', constant_values=0)
            dx = np.zeros((img.shape[0], dy_res.shape[1] - weight.shape[0] + 1, dy_res.shape[1] - weight.shape[0] + 1, weight.shape[2]))
            dx_temp = np.zeros((img.shape[0], dy_res.shape[1] - weight.shape[0] + 1, dy_res.shape[1] - weight.shape[0] + 1, weight.shape[3]))
            
            w = dx.shape[1]
            h = dx.shape[2]
            
            for c in range(weight.shape[2]):
                dx_temp = np.zeros((img.shape[0], dy_res.shape[1] - weight.shape[0] + 1, dy_res.shape[1] - weight.shape[0] + 1, weight.shape[3]))
                for i in range(weight.shape[0]):
                    for j in range(weight.shape[0]):
                        dx_temp = dx_temp + dy_res[:,i:i+w, j:j+h,:].copy() * weight[(k_size-1)-i, (k_size-1)-j,c,:]
                
                dx_temp = np.sum(dx_temp, axis = 3)
                dx[:,:,:,c] = dx_temp
            
            self.x.grad = self.x.grad + dx
        #pass
                
            
        if self.k in ops or self.k in params:
            img = self.x.top
            dy = self.grad
            
            dk = np.zeros((img.shape[1]-dy.shape[1]+1, img.shape[1]-dy.shape[1]+1, img.shape[3], dy.shape[3]))
            dk_temp = np.zeros((img.shape[1]-dy.shape[1]+1, img.shape[1]-dy.shape[1]+1,img.shape[3], img.shape[0]))
            
            w = dk.shape[0]
            h = dk.shape[1]
            k_size = dy.shape[1]
            
            # print(k_size)
            
            for c in range(dy.shape[3]):
                dk_temp = np.zeros((img.shape[1]-dy.shape[1]+1, img.shape[1]-dy.shape[1]+1,img.shape[3], img.shape[0]))
                for i in range(dy.shape[1]):
                    for j in range(dy.shape[1]):
                        temp = img[:,i:i+w, j:j+h,:]
                        temp_new = np.reshape(temp, (temp.shape[1], temp.shape[2], temp.shape[3], temp.shape[0]))
                        
                        new_temp =  temp_new.copy() * dy[:,(k_size-1)-i, (k_size-1)-j,c]
                        dk_temp = dk_temp + new_temp
                        
                
                dk_temp = np.sum(dk_temp, axis = 3)
                dk[:,:,:,c] =  dk_temp
                
            self.k.grad = self.k.grad + dk  
                
            #pass
