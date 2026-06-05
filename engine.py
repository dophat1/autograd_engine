import math
import random

"""
 All numbers in the network will be stored as Value class. The Value must stored:
 1. The real number of the value in the network
 2. The gradient at that value (default 0.0)
 3. The number get fed into it before (its children)
 4. The operator made up that value

 Each node stores its children and operation so that during the backward pass, it can compute the local gradients and pass them back to its children.
That's the whole engine. Every node knows:

- What created it
- Who fed into it
- So it can propagate gradients backwards
"""
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        data = self.data + other.data
        _children =     {self, other}
        _op = '+'
        out = Value(data, _children, _op)
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        data = self.data * other.data
        _children = {self, other}
        _op = '*'
        out = Value(data, _children, _op)
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
            

    def tanh(self):
        data = (math.e ** self.data - math.e ** (-self.data)) / (math.e ** self.data + math.e ** (-self.data))
        _children = {self}
        _op = 'tanh'
        out = Value(data, _children, _op)

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        self.grad = 1.0
        visited = set()
        topo = []
        def build_topo(v):      
            if v not in visited:
                for child in v._children:
                    build_topo(child)
                topo.append(v)
                visited.add(v)
            return topo
        topo = build_topo(self)
        for node in reversed(topo):
            node._backward()


    def __repr__(self):
        return f"Value(data={self.data})"



class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        out = []
        for neuron in self.neurons:
            neuron_out = neuron(x)
            out.append(neuron_out)
        return out
    
    def parameters(self):
        neuron_in_layer = []
        for neuron in self.neurons:
            neuron_in_layer += neuron.parameters()
        return neuron_in_layer

class Multi_Layer_Perceptron:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        shape = zip(sizes[:-1], sizes[1:])
        multi_layer = []
        for nini, nouti in shape:
            multi_layer.append(Layer(nini, nouti))
        self.layers = multi_layer
    
    def __call__(self, x):  
        for layer in self.layers:
            x = layer(x)
        if len(x) == 1:
            return x[0]
        else:
            return x
    
    def parameters(self):
        neuron_in_network = []
        for layer in self.layers:
            neuron_in_network += layer.parameters()
        
        return neuron_in_network



xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # targets


n = Multi_Layer_Perceptron(3, [4, 4, 1])


n.parameters()

for i in range(20):
    # y_pred using the n as neural network
    y_pred = [n(x) for x in xs]

    # Calculate all the loss squared
    loss = sum([(yp + -1 * yt)*(yp + -1 * yt) for yp, yt in zip(y_pred, ys)])

    # Backpropagation
    loss.backward() 
    print(f"The loss is  {loss} \n")

    # Updating weights and biases of all neurons in the network
    for param in n.parameters():
        param.data = param.data - param.grad * 0.01

    # After updating, reset the gradient of all weights and biases to 0
    for param in n.parameters():
        param.grad = 0