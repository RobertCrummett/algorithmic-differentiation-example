import algorithmicdifferentiation as ad

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import math, random
import numpy as np

random.seed(12345)

iris = datasets.load_iris()
x_train = iris['data']
x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
y_train = np.zeros((x_train.shape[0], 3))
for i, y in enumerate(iris['target']):
    y_train[i][y] = 1.0

class SoftmaxRegression:
    def __init__(self, input_shape, num_classes, l2, epochs, batch_size, base_lr):
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._epochs = epochs
        self._batch_size = batch_size
        self._base_lr = base_lr

        # these variables are just placeholders for the input features
        # of individual input sample
        self._inputs = [ad.Var(f'x_{i}') for i in range(input_shape)]

        # create the parameters and the pre-softmax outputs for each class
        self._params, self._biases, self._logits, self._labels = [], [], [], []
        for i in range(num_classes):
            # parameters needed to compute z_i
            b = ad.Var(f'b_{i}', value=0)
            ws = [
                    ad.Var(f'w_{i}_{j}', value=random.gauss(0, 0.1))
                    for j in range(input_shape)
                ]

            # output of the linear model
            z_i = b + sum(w * x for w, x in zip(ws, self._inputs))

            # placeholder for the true label to compute loss
            y = ad.Var(f'y_{i}')

            self._params.extend(ws)
            self._params.append(b)
            self._logits.append(z_i)
            self._labels.append(y)
        
        # compute the softmax output
        softmax_norm = sum(ad.Exp(m) for m in self._logits)
        self._outputs = [ad.Exp(l) / softmax_norm for l in self._logits]

        # and the cross entropy loss
        self._loss = -sum(
                l * ad.Log(o + 1e-6) for l, o in zip(self._labels, self._outputs)
                ) + l2 * sum(w * w for w in self._params)

        # finally, compute the gradients for each parameter
        self._grads = [self._loss.backward(p).simplify() for p in self._params]

    def fit(self, X, y):
        ''' train the model on the given data '''
        history = []
        batch_count = 0
        for epoch in range(self._epochs):
            idx = np.random.permutation(X.shape[0])
            for i in range(0, len(X), self._batch_size):
                batch_idx = idx[i:i+self._batch_size]
                loss = self._sgd_step(
                        X[batch_idx], y[batch_idx],
                        lr=self._base_lr / math.sqrt(1 + epoch)
                        )
                history.append(loss)

            if epoch % 10 == 0:
                print(f'epoch: {epoch}\tloss: {loss:.3f}')

        return history

    def _sgd_step(self, batch_x, batch_y, lr):
        ''' perform one step of stochastic gradient descent '''

        # here we accumulate gradients and loss
        grad_acc = [0.0] * len(self._grads)
        loss_acc = 0.0
        for x, y in zip(batch_x, batch_y):
            # set input and output placeholders to the values of this sample
            for input_, value in zip(self._inputs, x):
                input_.value = value
            for label_, value in zip(self._labels, y):
                label_.value = value

            # accumulate loss and gradients
            loss_acc += self._loss.compute()
            for i, g in enumerate(self._grads):
                grad_acc[i] += g.compute()

        # average the gradients and modify the parameters
        for p, g in zip(self._params, grad_acc):
            p.value = p.value - lr * g / len(batch_x)

        # return average loss for this minibatch
        return loss_acc / len(batch_x)

    def predict_proba(self, batch_x):
        ''' predict the class probabilities for each input sample in a batch '''
        preds = []
        for x in batch_x:
            # set input placeholders
            for input_, value in zip(self._inputs, x):
                input_.value = value

            # compute output probability for each class
            preds.append([o.compute() for o in self._outputs])

        return preds

    def predict(self, batch_x):
        ''' predict the class of each output sample '''
        probs = self.predict_proba(batch_x)
        return [
                max(enumerate(ps), key=lambda x: x[1])[0]
                for ps in probs
                ]

model = SoftmaxRegression(
        input_shape=len(x_train[0]), num_classes=len(y_train[0]),
        l2=1e-6, epochs=100, batch_size=32, base_lr=5e-2
    )
history = model.fit(x_train, y_train)

plt.plot(history)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.show()

preds = model.predict(x_train)

from sklearn.metrics import confusion_matrix
print()
print(confusion_matrix(iris['target'], preds))
print((preds == iris['target']).mean())
