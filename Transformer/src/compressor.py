import collections
import gzip
import os
import time
import utils
import struct
import json
from absl import app
from absl import flags
from absl import logging
import shutil

import numpy as np
import torch
import torch.nn.functional as F

import compress_model
import arithmeticcoding_fast
import utils

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
torch.set_printoptions(profile="full") 
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_integer('batch_size', 512, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Adam Optimizer learning rate.')
flags.DEFINE_integer('hidden_dim', 256, 'Feature dimension.')
flags.DEFINE_integer('vocab_dim', 64, 'Feature dimension.')
flags.DEFINE_integer('n_layers', 1, 'Number of Attention layers.')
flags.DEFINE_integer('ffn_dim', 4096, 'MLP dimension in model.')
flags.DEFINE_integer('n_heads', 8, 'Number of heads for attention.')
flags.DEFINE_string(
    'feature_type', 'sqr',
    'Nonlinearity function for feature. Can be relu, elu+1, sqr, favor+, or favor+{int}.'
)
flags.DEFINE_enum(
    'compute_type', 'iter', ['iter', 'ps', 'parallel_ps'],
    'Which type of method to compute: iter = iterative algorithm, ps = implementation using torch.cumsum, parallel_ps = implementation using custom log prefix sum implementation.'
)
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay for regularization.')

# Training parameters
flags.DEFINE_string('gpu_id', '0', 'ID of GPU.')
flags.DEFINE_integer('random_seed', 0, 'Random seed for both Numpy and Torch.')
flags.DEFINE_integer('print_step', 1000, 'Interval to print metrics.')
# Dataset parameters
flags.DEFINE_integer('seq_len', 8, 'Maximum sequence length (L).')
flags.DEFINE_integer('vocab_size', 256, 'Vocabulary size of data.')
flags.DEFINE_string('input_dir', '../data/', 'input data dir')
flags.DEFINE_string('prefix', 'text8', 'output dir')

def encode(temp_dir, compressed_file, FLAGS, series, train_data, last_train_data):
  
  bs = FLAGS.batch_size
  print(temp_dir)
  print(compressed_file)
  print(temp_dir+"/"+compressed_file)
  f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'wb') for i in range(bs)]
  bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
  enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]
  
  prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
  cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
  cumul[1:] = np.cumsum(prob*10000000 + 1)
  
  iter_num = len(train_data) // FLAGS.batch_size
  ind = np.array(range(bs))*iter_num
  iter_num -= FLAGS.seq_len

  for i in range(bs):
    for j in range(FLAGS.seq_len):
      enc[i].write(cumul, series[ind[i]+j])
  
  cumul_batch = np.zeros((bs, FLAGS.vocab_size+1), dtype = np.uint64)

  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)
  print(np.random.rand(4))

  model = compress_model.SLiMPerformer(FLAGS.vocab_size, FLAGS.vocab_dim, FLAGS.hidden_dim,
                                             FLAGS.n_layers, FLAGS.ffn_dim,
                                             FLAGS.n_heads, FLAGS.feature_type, FLAGS.compute_type) #.cuda()
  print(model)
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
  print(iter_num)
  for train_index in range(iter_num):
    model.train()
    train_batch = train_data[ind, :]
    y = train_batch[:, -1]
    train_batch = torch.from_numpy(train_batch).long() #.cuda().long()
    
    train_loss, logits = model.full_loss(train_batch, with_grad=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    logits = logits.transpose(1, 2)
    prob = logits[:, -1, :]
    prob = F.softmax(prob, dim=1).detach().cpu().numpy()
    cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
    
    for i in range(bs):
      enc[i].write(cumul_batch[i,:], y[i])
    
    ind += 1
    if train_index % FLAGS.print_step == 0:
      size = 0
      for cf in os.listdir(temp_dir):
        size += os.path.getsize(temp_dir+"/"+cf)
      print(train_index, ":", train_loss.item()/np.log(2), "size:", size/(1024*1024))
  
  for i in range(bs):
    enc[i].finish()
    bitout[i].close()
    f[i].close()

  if last_train_data is not None:
    print("last series")
    f = open(temp_dir+"/"+compressed_file+'.last','wb')
    bitout = arithmeticcoding_fast.BitOutputStream(f)
    enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
    prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
    cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)
  
    for j in range(len(last_train_data)):
      enc.write(cumul, last_train_data[j])
    print("Last encode part don't need inference.")
  
    enc.finish()
    bitout.close()
    f.close()
  
  return
    
def var_int_encode(byte_str_len, f):
  while True:
    this_byte = byte_str_len&127
    byte_str_len >>= 7
    if byte_str_len == 0:
      f.write(struct.pack('B',this_byte))
      break
    f.write(struct.pack('B',this_byte|128))
    byte_str_len -= 1

def main(_):

  # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

  base_dir = "..\\data\\compressed"
  temp_dir = os.path.join(base_dir, "{}_{}_{}_{}_bs{}_{}_seq{}_temp".format(FLAGS.prefix, FLAGS.vocab_dim, FLAGS.hidden_dim, FLAGS.ffn_dim, FLAGS.batch_size, FLAGS.n_layers, FLAGS.seq_len))
  compressed_file = os.path.basename(temp_dir.replace("_temp", ".compressed"))
  os.makedirs(temp_dir, exist_ok=True)
  print(compressed_file)
  
  def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))
  
  old_seq_len = FLAGS.seq_len
  FLAGS.seq_len = FLAGS.seq_len*(FLAGS.hidden_dim // FLAGS.vocab_dim)
  print("FLAGS.seq_len change from {} to {} due to FLAGS.vocab_dim = {} and FLAGS.hidden_dim = {}.".format(old_seq_len, FLAGS.seq_len, FLAGS.vocab_dim, FLAGS.hidden_dim))
  
  with open(FLAGS.input_dir, 'rb') as fp:#, encoding='latin-1') as fp:
    series = np.fromstring(fp.read(), dtype=np.uint8)
  train_data = strided_app(series, FLAGS.seq_len+1, 1)

  total_length = len(train_data)
  if total_length % FLAGS.batch_size == 0:
    encode(temp_dir, compressed_file, FLAGS, series, train_data, None)
  else:
    l = total_length // FLAGS.batch_size * FLAGS.batch_size
    encode(temp_dir, compressed_file, FLAGS, series[:l+FLAGS.seq_len], train_data[:l], series[l:])
  
  #Combined compressed results
  f = open(base_dir+"\\"+compressed_file+'.combined','wb')
  for i in range(FLAGS.batch_size):
    f_in = open(temp_dir+'/'+compressed_file+'.'+str(i),'rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
  
  if total_length % FLAGS.batch_size != 0:
    f_in = open(temp_dir+'/'+compressed_file+'.last','rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
  f.close()
  
  total = 0
  for ff in os.listdir(temp_dir):
    total += os.path.getsize(temp_dir+'/'+ff)
  
  print(total/(1024*1024))

  len_series = len(series)
  output_data = {'len_series': len_series}

  prefix_basename = os.path.basename(FLAGS.prefix)
    # Remove the file extension
  prefix_name, _ = os.path.splitext(prefix_basename)

  with open(base_dir+"\\"+prefix_name+'_transformer.json', 'w') as json_file:
    json.dump(output_data, json_file)
  
  #Remove temp file
  shutil.rmtree(temp_dir)
  

if __name__ == '__main__':
  app.run(main)
