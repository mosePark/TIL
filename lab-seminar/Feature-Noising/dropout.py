import sys
sys.path.insert(0, '..')

import d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn

def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    # In this case, all elements are dropped out
    if drop_prob == 1:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) > drop_prob
    return mask * X / (1.0-drop_prob)

X = nd.arange(16).reshape((2, 8))
print(dropout(X, 0))
print(dropout(X, 0.5))
print(dropout(X, 1))

# 출처 : https://ko.d2l.ai/chapter_deep-learning-basics/dropout.html
# 참고 논문 : Dropout: a simple way to prevent neural networks from overfitting, N Srivastava, G Hinton, A Krizhevsky, I Sutskever, R Salakhutdinov

## 추후 목표 : 코드를 내 방식대로 좀 더 다듬어보기.
