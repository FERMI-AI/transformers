import random
import numpy as np
from tinygrad.state import get_parameters
from tinygrad.nn.optim import Adam
from extra.training import train, evaluate
from models.transformer import Transformer

docs = """
    Modeling transformers with tinygrad (does it have a tranformer block thingy ?)
    - turns out they do have it 
    - takes in - (syms, maxlen, layers, embed_dim, num_heads, ff_dim)
  """

def make_dataset():
  ds = []
  for i in range(100):
    for j in range(100):
      s = i+j
      ds.append([i//10, i%10, j//10, j%10, s//100, (s//10)%10, s%10])
  random.shuffle(ds)
  ds = np.array(ds).astype(np.float32)
  ds_X = ds[:, 0:6]
  ds_Y = np.copy(ds[:, 1:])
  ds_X_train, ds_X_test = ds_X[0:8000], ds_X[8000:]
  ds_Y_train, ds_Y_test = ds_Y[0:8000], ds_Y[8000:]
  return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test

model = Transformer(10,6,2,128,4,32)
X_train, Y_train, X_test, Y_test = make_dataset()
lr = .03
for i in range(10):
  optim = Adam(get_parameters(model), lr=lr)
  train(model, X_train, Y_train, optim, 50, BS=64)
  acc, Y_test_preds = evaluate(model, X_test, Y_test, num_classes = 10, return_predict=True)
  print(acc)

