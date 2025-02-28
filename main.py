from abc import ABC, abstractmethod

class DifferentiableSymbolicOperation(ABC):
    @abstractmethod
    def backward(self, var):
        pass

    @abstractmethod
    def compute(self):
        pass

class Const(DifferentiableSymbolicOperation):
    def __init__(self, value):
        self.value = value
    
    def backward(self, var):
        return Const(0)

    def compute(self):
        return self.value

    def __repr__(self):
        return str(self.value)

class Var(DifferentiableSymbolicOperation):
    def __init__(self, name, value=None):
        self.name, self.value = name, value

    def backward(self, var):
        return Const(1) if self == var else Const(0)

    def compute(self):
        if self.value is None:
            raise ValueError('unassigned variable')
        return self.value
    
    def __repr__(self):
        return f'{self.name}'

class Sum(DifferentiableSymbolicOperation):
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def backward(self, var):
        return Sum(self.x.backward(var), self.y.backward(var))

    def compute(self):
        return self.x.compute() + self.y.compute()

    def __repr__(self):
        return f'({self.x} + {self.y})'

class Mul(DifferentiableSymbolicOperation):
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def backward(self, var):
        return Sum(
                Mul(self.x.backward(var), self.y),
                Mul(self.x, self.y.backward(var))
            )

    def compute(self):
        return self.x.compute() * self.y.compute()

    def __repr__(self):
        return f'({self.x} * {self.y})'

def simplify(node):
    if isinstance(node, Sum):
        return simplify_sum(node)
    elif isinstance(node, Mul):
        return simplify_mul(node)
    else:
        return node

def simplify_sum(node):
    # first, recursively simplify the children
    x = simplify(node.x)
    y = simplify(node.y)

    x_const = isinstance(x, Const)
    y_const = isinstance(y, Const)

    if x_const and y_const:
        # propogate constants
        return Const(x.value + y.value)
    elif x_const and x.value == 0:
        # 0 + y = y
        return y
    elif y_const and y.value == 0:
        # x + 0 = x
        return x
    else:
        # return a new node with the simplified operands
        return Sum(x, y)

def simplify_mul(node):
    # first, recursively simplify the children
    x = simplify(node.x)
    y = simplify(node.y)

    x_const = isinstance(x, Const)
    y_const = isinstance(y, Const)

    if x_const and y_const:
        # propogate constants
         return Const(x.value * y.value)
    elif x_const and x.value == 0:
        # 0 * y = 0
        return Const(0)
    elif x_const and x.value == 1:
        # 1 * y = y
        return y
    elif y_const and y.value == 0:
        # x * 0 = 0
        return Const(0)
    elif y_const and y.value == 1:
        # x * 1 = x
        return x
    else:
        # return a new node with the simplified operands
        return Mul(x, y)

if __name__ == '__main__':
    x = Var('x', 3)
    y = Var('y', 2)
    z = Sum(Sum(Mul(x,x), Mul(Const(3), Mul(x,y))), Const(1))

    print(z)
    print(z.compute())
    print(z.backward(x).compute())
    print(z.backward(x).backward(y))

    print()
    print(Sum(x, Const(0)))
    print(simplify_sum(Sum(x, Const(0))))

    print()
    print(Sum(Const(2), Const(3)))
    print(simplify_sum(Sum(Const(2), Const(3))))

    print()
    print(z.backward(x))
    print(simplify(z.backward(x)))

    print()
    print(type(simplify(z.backward(x).backward(y))))
