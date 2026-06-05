import math
import random

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None
    
    def __add__(self, other):
        data = self.data + other.data
        _children =     {self, other}
        _op = '+'
        out = Value(data, _children, _op)
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
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
        act = sum((wi * Value(xi) for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        out = []
        for neuron in self.neurons:
            neuron_out = neuron(x)
            out.append(neuron_out)
        return out

a = Layer(3, 4)
print(a([1,2,3]))
