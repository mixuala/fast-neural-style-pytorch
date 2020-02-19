
import tensorflow as tf
import numpy as np
import PIL.Image

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

def sq_center_crop(filename, size=256, normalize=True):
  """
  return the largest square center crop from image, resized to (size,size)
  
  Returns:
    image tensor of domain=(0.,1.) if normalize==True else domain(0,255)
  """
  import torchvision
  parts = tf.strings.split(filename, '/')
  label = parts[-2]
  pil_image = tf.keras.preprocessing.image.load_img(filename)
  pil_image = torchvision.transforms.Resize(size)(pil_image)
  pil_image = torchvision.transforms.CenterCrop(size)(pil_image)
  # torch_tensor = np_array( torchvision.transforms.ToTensor()(pil_image) ) # CHANNELS_FIRST
  if normalize:
    image = tf.keras.preprocessing.image.img_to_array(pil_image, dtype=float)  # 255. CHANNELS_LAST
    image = tf.convert_to_tensor(image/255., dtype=tf.float32)
  else:
    image = tf.keras.preprocessing.image.img_to_array(pil_image, dtype=int)  # 255 CHANNELS_LAST
    image = tf.convert_to_tensor(image, dtype=tf.uint8)
  return image, label

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

# dataset helpers
def dataset_size(dataset):
  return dataset.reduce(0, lambda x, _: x + 1).numpy()


def torch_transforms(filename):
  """
  same as pytorch implementation, but returns CHANNELS_LAST tf.tensor()
  WARN: doesn't work with Dataset.map()  
  """
  import torchvision
  parts = tf.strings.split(filename, '/')
  label = parts[-2]
  print(tf.is_tensor(filename), )
  if tf.executing_eagerly():
    filename = filename.numpy().decode()
  elif isinstance( filename, tf.Tensor):
    # Dataset.map(), does not execute eagerly
    assert False, "error: cannot convert Tensor dtype=string to python string"
    
  pil_image = tf.keras.preprocessing.image.load_img(filename)
  pil_image = torchvision.transforms.Resize(TRAIN_IMAGE_SIZE)(pil_image)
  pil_image = torchvision.transforms.CenterCrop(TRAIN_IMAGE_SIZE)(pil_image)
  image = np.array( torchvision.transforms.ToTensor()(pil_image) ) # CHANNELS_FIRST
  image = tf.transpose(image, perm=[1,2,0]) # normalized, CHANNELS_LAST
  # image *= 255. # 255, CHANNELS_LAST 
  return image, label

def batch_torch_transforms(list_ds, batch_size=None):
  """
  transform a Dataset list of image filenames in to a batch of resized images, labels
  uses torchvision.transforms.Resize() to crop largest square before resize

  WARNING: does NOT lazy load from Dataset

  Args:
    list_ds: 
      tf.data.Dataset.list_files('{}/*/*'.format(DATASET_PATH)), or
      next(iter(list_ds.batch(BATCH_SIZE))), batch of string tensors


  usage:
    train_dataset = batch_torch_transforms(list_ds.take(10), BATCH_SIZE) # NOT lazy loaded
    for filenames in list_ds.batch(BATCH_SIZE):
      x_train, y_true = batch_torch_transforms(filenames, BATCH_SIZE)

  """
  def _process_batch(filenames):
    images = []
    labels = []
    for i, filename in enumerate(filenames):
      image, label = torch_transforms(filename)
      images += [image]
      labels += [label]
      if batch_size is not None and i+1>=batch_size:
        break
    images = tf.stack(images, axis=0)
    labels = tf.stack(labels, axis=0)
    return (images, labels)

  if isinstance(list_ds, tf.data.Dataset):
    if batch_size is not None:
      list_ds = list_ds.batch(batch_size)
    batch = []  
    for filenames in list_ds:
      batch.append( _process_batch(filenames) ) # x_train, y_true
    return batch

  if isinstance(list_ds, tf.Tensor):
    filenames = [f.decode() for f in list_ds.numpy()]
    return _process_batch(filenames)