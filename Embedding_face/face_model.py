from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import mxnet as mx
from sklearn.preprocessing import normalize


def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  #sym = all_layers[layer]
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    if self.args.device == "gpu":
      ctx = mx.gpu(self.args.gpu_id)
    else:
      ctx = mx.cpu(self.args.cpu_id)
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    if len(args.model_insight)>0:
      self.model = get_model(ctx, image_size, args.model_insight, 'fc1')
    
    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size

  def get_feature(self, aligned):
    #input_blob=aligned # process with batch
    input_blob = np.expand_dims(aligned, axis=0) # process without batch
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    #embedding = normalize(embedding).flatten()
    #embedding = sklearn.preprocessing.normalize(embedding).flatten()
    
    return embedding

