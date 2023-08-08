from tinygrad.tensor import Tensor
import tinygrad.nn as nn 
from models.transformer import Transformer

docs = """
    Modeling transformers with tinygrad (does it have a tranformer block thingy ?)
    - turns out they do have it 
    - takes in - (syms, maxlen, layers, embed_dim, num_heads, ff_dim)
  """

model = Transformer(10,6,2,128,4,32)
