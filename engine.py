class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._children = set(_children)
        self._op = _op
    
    def __add__(self, other):
        data = self.data + other.data
        _children =     {self, other}
        _op = '+'
        out = Value(data, _children, _op)
        return out
    
    def __mul__(self, other):
        data = self.data * other.data
        _children = {self, other}
        _op = '*'
        out = Value(data, _children, _op)
        return out
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
