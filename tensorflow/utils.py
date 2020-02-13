
import tensorflow as tf
import numpy as np

# Gram Matrix
def gram(tensor):
  """
  same as torch gram(), but uses CHANNELS_LAST format
  tested OK vs torch_gram()
  """
  B, H, W, C = tensor.shape
  x = tf.reshape( tensor, (B, -1, C) )
  return  tf.matmul(x, x, transpose_a=True) / (C*H*W)

def gram_matrix(array, normalize_magnitude=True):
  (b,*_,c) = array.shape
  b = tf.shape(array)[0]
  array_flat = tf.reshape(array, shape=[b, -1, c])
  gram_matrix = tf.matmul(array_flat, array_flat, transpose_a=True)
  if normalize_magnitude:
    length = tf.shape(array_flat)[-2]
    # ???: normalize magnitude with H*W*C ???
    gram_matrix /= tf.cast(length *c , tf.float32) 
  return gram_matrix

import torch
def torch_gram(ch_last_array):
  ch_first_array = tf.transpose(ch_last_array, perm=[0,3,1,2]).numpy()
  tensor = torch.tensor(ch_first_array)
  B, C, H, W = tensor.shape
  x = tensor.view(B, C, H*W)
  x_t = x.transpose(1, 2)
  return  torch.bmm(x, x_t) / (C*H*W)


def vgg_input_preprocess(image, max_dim=512, **kvargs):
  """ apply appropriate VGG preprocessing to image, shape=(?,h,w,c)

  should be same as `tf.keras.applications.vgg19.preprocess_input(rgb_data)/255.0`

  NOTE: 
    - x_train from dataset will be dtype=tf.float32, domain=(0,1.) AFTER tf.image.convert_image_dtype(img, tf.float32)
    - otherwise image could be tf.float32 or tf.uint8

  Args:
    image: tensor, accepts domain=(0.,1.) or dtype=tf.uint8, domain=(0,255)
  Returns: normalized, mean-centered, and BGR ordered image tensor in batch form, shape=(1, h, w, c)
    domain=(0.,1.), dtype=tf.float32
  """
  # normalize, domain (0,1)
  assert len(image.shape)==4, "expecting shape=(b,h,w,c)"
  x = tf.image.convert_image_dtype(image, tf.float32)
  
  # expecting domain=(0.,1.)
  output = tf.keras.applications.vgg19.preprocess_input(x*255.0)/255.0
  return output


def random_sq_crop(image, size=256, margin_pct=5):
  """
  take a square crop from image of size `margin_pct` smaller (e.g. 5% smaller) than the short dimension
  and randomly offset from center. Resize crop to return a square image of dim=size

  NOTE: does NOT work inside tf.data.Dataset.map() because image.getShape() is (None, None, None)
  """
  scale = 1. + margin_pct/100
  h,w,c = image.shape
  min_dim = np.amin([h,w])
  crop_size = np.amax([256, min_dim//scale] ).astype(int)
  offset = (np.random.random(size=2) * (min_dim - crop_size)).astype(int)
  # min_dim = tf.minimum(h,w)
  # crop_size = tf.maximum(size, tf.cast( tf.cast(min_dim,tf.float32)/scale, tf.int32) )
  # offset = tf.random.uniform(shape=[2], maxval=(min_dim - crop_size), dtype=tf.int32)
  if w==min_dim: # portrait
    (x,y) = offset[0], offset[1]+((h-crop_size)//2)
  else: # landscape
    (x,y) = offset[0]+((w-crop_size)//2), offset[1]
  image = tf.image.crop_to_bounding_box( image, y, x, crop_size, crop_size)
  image = tf.image.resize(image, (size,size) )
  return image