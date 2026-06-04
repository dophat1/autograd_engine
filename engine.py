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


a = Value(2.0)
b = Value(3.0)
c = a * b
print(c)
c.backward()
print(c.grad)
print(a.grad)
print(b.grad)