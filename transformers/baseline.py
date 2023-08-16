import random
import numpy as np
from tinygrad.state import get_parameters
from tinygrad.nn.optim import Adam
from extra.training import train, evaluate
from models.transformer import Transformer
from tqdm import trange
from tinygrad.tensor import Tensor, Device

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

def sparse_categorical_crossentropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten().astype(np.int32)
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = make_dataset()
  model = Transformer(10,6,2,128,4,32)
  lr = .03
  BS = 128
  optim = Adam(get_parameters(model), lr=lr)
  lossfn = sparse_categorical_crossentropy
  for j in (t:=trange(10)):
    for i in range(X_train.shape[0]//BS):
      samp = np.random.randint(0, X_train.shape[0], size=(BS))
      x = Tensor(X_train[samp])
      y = Y_train[samp]
      out = model.forward(x)
      loss = lossfn(out, y)
      optim.zero_grad()
      loss.backward()
      optim.step()
    lr /= 1.2
    t.set_description(f"loss : {loss.detach().cpu().numpy()}")


