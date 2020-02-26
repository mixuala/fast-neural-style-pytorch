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

def batch_reduce_sum(lossFn, y_true, y_pred, weight=1.0, name=None):
  batch_size = y_pred[0].shape[0] if isinstance(y_pred, (tuple,list)) else y_pred.shape[0]
  losses = tf.zeros(batch_size)
  for a,b in zip(y_true, y_pred):
    # batch_reduce_sum()
    # loss = tf.keras.losses.MSE(a,b)
    loss = lossFn(a,b)
    loss = tf.reduce_sum(loss, axis=[i for i in range(1,len(loss.shape))] )
    losses = tf.add(losses, loss)
  if name is not None: name = "{}_loss".format(name) 
  return tf.multiply(losses, weight, name=name) # shape=(b,)  


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
  h,w,c = image.shape
  min_dim = np.amin([h,w])
  if min_dim<=size:
    crop_size = min_dim
  else: 
    scale = 1. + margin_pct/100
    crop_size = np.amax([ size, min_dim//scale] ).astype(int)
  offset = (np.random.random(size=2) * (min_dim - crop_size)).astype(int)
  if w==min_dim: # portrait
    (x,y) = offset[0], offset[1]+((h-crop_size)//2)
  else: # landscape
    (x,y) = offset[0]+((w-crop_size)//2), offset[1]

  if (x+crop_size > w):
    x -= (x+crop_size) - w
  
  if (y+crop_size > h):
    y -= (y+crop_size) - h

  image = tf.image.crop_to_bounding_box( image, y, x, crop_size, crop_size)
  image = tf.image.resize(image, (size,size) )
  return image  


# dataset helpers
def dataset_size(dataset):
  return dataset.reduce(0, lambda x, _: x + 1).numpy()


def torch_transforms(filename, size=256):
  """
  same as pytorch implementation, but returns CHANNELS_LAST tf.tensor()
  WARN: doesn't work with Dataset.map()  
  """
  import torchvision
  parts = tf.strings.split(filename, '/')
  label = parts[-2]
  if tf.is_tensor(filename):
    if tf.executing_eagerly():
      filename = filename.numpy().decode()
    else:
      # Dataset.map(), does not execute eagerly
      assert False, "error: cannot convert Tensor dtype=string to python string"
    
  pil_image = tf.keras.preprocessing.image.load_img(filename)
  pil_image = torchvision.transforms.Resize(size)(pil_image)
  pil_image = torchvision.transforms.CenterCrop(size)(pil_image)
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


class ImageRecordDatasetFactory():
  """
  https://tensorflow.google.cn/tutorials/load_data/tfrecord#walkthrough_reading_and_writing_image_data

  create a TFRecordDataset of images from folder tree

  usage:
    # write recordset
    list_ds = tf.data.Dataset.list_files('{}/*/*'.format(DATASET_PATH)).take(LIMIT)
    maker = ImageRecordDatasetFactory()
    recordsetpath = maker.write(recordsetpath, list_ds)
    print( "file={}, size={}".format(recordsetpath, os.path.getsize(recordsetpath)) )

    # read recordset
    rec_dataset = tf.data.TFRecordDataset(recordsetpath)
    image_ds = rec_dataset.map(maker.example_parser(image2tensor=True)).take(10)
    for d in image_ds:
      image_tensor = d['image']
      break
  """
  def _int64_feature(self, value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  def _bytes_feature(self, value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  def _float_feature(self, value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  def image_example(self, image_string, label=b''):
    """
    Args:
      image_string = tf.io.read_file(filename)
    """
    image_shape = tf.image.decode_jpeg(image_string).shape
    label = bytes(label)
    feature = {
        'h': self._int64_feature(image_shape[0]),
        'w': self._int64_feature(image_shape[1]),
        'c': self._int64_feature(image_shape[2]),
        'image_raw': self._bytes_feature(image_string),
        'label': self._bytes_feature(label)
    }      
    return tf.train.Example(features=tf.train.Features(feature=feature))


  def write(self, savepath, list_ds, label=b'', square=False):
    """ write a tf.data.TFRecordDataset to the local filesystem

    Args:
      savepath: path with '.tfrecord' extension
      list_ds: tf.data.Dataset.list_files(images)
      label: label='folder' uses dirname as label
    """
    if not str(savepath).endswith('.tfrecord'):
      savepath = '{}.tfrecord'.format(savepath)
    if not os.path.isdir(os.path.dirname(savepath)):
      os.mkdir(os.path.dirname(savepath))
    with tf.io.TFRecordWriter(savepath) as writer:
      for item in list_ds:
        if isinstance(item, (tuple, list)):
          filename, label = item
        elif label=="folder":
          filename = item
          label = os.path.basename(os.path.dirname('./dataset/coco10.tfrecord'))
        else:
          filename = item
        # image_string = open(filename, 'rb').read()
        image_string = tf.io.read_file(filename)

        # process before write
        if square:
          image_string = self.crop_square(image_string)

        tf_example = self.image_example(image_string, label=label)
        writer.write(tf_example.SerializeToString())
      return savepath


  def example_parser(self, image2tensor=True):
    image_feature_description = {
      'h': tf.io.FixedLenFeature([], tf.int64),
      'w': tf.io.FixedLenFeature([], tf.int64),
      'c': tf.io.FixedLenFeature([], tf.int64),
      'image_raw': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.string),
    }
    def _parser(example_proto):
      # Parse the input tf.Example proto using the dictionary above.
      parsed =  tf.io.parse_single_example(example_proto, image_feature_description)
      if image2tensor:
        parsed['image'] = self.image2tensor(parsed['image_raw'])
        del parsed['image_raw']
      return parsed
    return _parser

  @staticmethod
  def image2tensor(image_raw):
    # image = tf.image.decode_jpeg(image_raw)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

  @staticmethod
  def crop_square(image_string, size=256):
    img = ImageRecordDatasetFactory.image2tensor(image_string)
    img = ImageRecordDatasetFactory.random_sq_crop(img, size)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    # print("img.shape", img.shape)
    return tf.io.encode_jpeg(img, quality=80)

  @staticmethod
  def random_sq_crop(image, size=256, margin_pct=5):
    """
    take a square crop from image of size `margin_pct` smaller (e.g. 5% smaller) than the short dimension
    and randomly offset from center. Resize crop to return a square image of dim=size

    NOTE: does NOT work inside tf.data.Dataset.map() because image.getShape() is (None, None, None)
    """
    return random_sq_crop(image, size, margin_pct)


class PerceptualLosses_Loss(tf.keras.losses.Loss):
  name="PerceptualLosses_Loss"
  reduction=tf.keras.losses.Reduction.AUTO
  RGB_MEAN_NORMAL_VGG = tf.constant( [0.48501961, 0.45795686, 0.40760392], dtype=tf.float32)
  batch_size = 4

  def __init__(self, loss_network, target_style_gram, batch_size=4, loss_weights=[1.,1.]):
    super(PerceptualLosses_Loss, self).__init__( name=self.name, reduction=self.reduction )
    self.target_style_gram = target_style_gram # repeated in y_true
    # print("PerceptualLosses_Loss init()", type(target_style_gram), type(self.target_style_gram))
    self.VGG = loss_network
    self.batch_size = batch_size # hack: y_pred.shape=(None, 256,256,3) when using model.fit()
    self.CONTENT_WEIGHT = loss_weights[0]
    self.STYLE_WEIGHT = loss_weights[1]

  def call(self, y_true, y_pred):
    # generated_batch = y_pred
    b,h,w,c = y_pred.shape
    #???: y_pred.shape=(None, 256,256,3), need batch dim for gram(value)
    generated_batch = tf.reshape(y_pred, (self.batch_size,h,w,c) )

    # generated_batch: expecting domain=(+-int), mean centered
    generated_batch = tf.nn.tanh(generated_batch) # domain=(-1.,1.), mean centered
    
    # reverse VGG mean_center
    generated_batch = tf.add( generated_batch, self.RGB_MEAN_NORMAL_VGG) # domain=(0.,1.)
    generated_batch_BGR_centered = tf.keras.applications.vgg19.preprocess_input(generated_batch*255.)/255.
    generated_content_features, generated_style_features = self.VGG( generated_batch_BGR_centered, preprocess=False )
    generated_style_gram = [ gram(value)  for value in generated_style_features ]  # list

    # y_pred = generated_content_features + generated_style_gram
    # print("PerceptualLosses_Loss: y_pred, output_shapes=", type(y_pred), [v.shape for v in y_pred])
    # PerceptualLosses_Loss: y_pred, output_shapes= [
    #   ([4, 16, 16, 512]), ([4, 64, 64]), ([4, 128, 128]), ([4, 256, 256]), ([4, 512, 512]), ([4, 512, 512])
    # ]

    if tf.is_tensor(y_true):
      # using: model.fit( x=xx_Dataset, )
      x_train = y_true
      x_train_BGR_centered = tf.keras.applications.vgg19.preprocess_input(x_train*255.)/255.
      target_content_features, _ = self.VGG(x_train_BGR_centered, preprocess=False )
      # ???: target_content_features[0].shape=(None, None, None, 512), should be shape=(4, 16, 16, 512)
      target_content_features = [tf.reshape(v, generated_content_features[i].shape) for i,v in enumerate(target_content_features)]

    elif isinstance(y_true, tuple):
      # using: model.fit( x=xy_true_Dataset, )
      # WARN: this branch fails with error when used with model.fit():
      # ValueError: Error when checking model target: the list of Numpy arrays that you are passing to your model is not the size the model expected. 
      #   Expected to see 1 array(s), for inputs ['output_1'] but instead 
      #   got the following list of 6 arrays: [
      #     <tf.Tensor 'args_1:0' shape=(None, 16, 16, 512) dtype=float32>, 
      #     <tf.Tensor 'args_2:0' shape=(None, 64, 64) dtype=float32>, 
      #     <tf.Tensor 'args_3:0' shape=(None, 128, 128) dtype=float32>, 
      #     <tf.Tensor 'arg...
      # print("detect y_true is tuple(target_content_features + self.target_style_gram)", y_true[0].shape)
      target_content_features = y_true[:len(generated_content_features)]
      if self.target_style_gram is None:
        self.target_style_gram = y_true[len(generated_content_features):]
    else:
      assert False, "unexpected result for y_true"

    # losses = tf.keras.losses.MSE(y_true, y_pred)
    lossFn = tf.keras.losses.MSE
    c_loss = batch_reduce_sum(lossFn, target_content_features, generated_content_features, self.CONTENT_WEIGHT, name='content_loss')
    s_loss = batch_reduce_sum(lossFn, self.target_style_gram, generated_style_gram, self.STYLE_WEIGHT, name='style_loss')
    return (c_loss, s_loss)



###
### dataset helpers
###
def xyGenerator255(image_ds, limit=None):
  """ returns (x_train, y_true) = (batch_image, batch_image)
  images scaled to domain=(0.,255.)
  """
  def gen():
    for d in image_ds:
      image = d['image']*255.
      yield (image, image)
  return gen 

def loadDataset(tfrecord_path):
  """get dataset of (256,256,3) images from `.tfrecord` file

  Return 
    (x_train, y_true) tuple of duplicate images
    images as h,w,c(RGB) domain=(0., 255.)
  """
  rec_dataset = tf.data.TFRecordDataset(tfrecord_path)
  recordsetMaker = ImageRecordDatasetFactory()
  image_ds = rec_dataset.map(recordsetMaker.example_parser(image2tensor=True))
  
  xx_Dataset255 = tf.data.Dataset.from_generator(
    generator=xyGenerator255(image_ds),
    output_types=(tf.float32, tf.float32),
    output_shapes=(
      (256,256,3), (256,256,3),
    ),
  )
  return xx_Dataset255


###
### loss helpers
###
MSELoss = tf.keras.losses.MSE
def get_content_loss(y_true_content, y_pred_content):
  content_loss = 0.
  for (y_true, y_pred) in zip(y_true_content, y_pred_content):
    batch_size = y_pred.shape[0]
    # WRONG!!!  content_loss += tf.reduce_sum(MSELoss(y_true[:batch_size], y_pred))/batch_size
    content_loss += tf.reduce_mean(MSELoss(y_true[:batch_size], y_pred))
  return content_loss

def get_style_loss(y_true_gram, y_pred_gram):
  style_loss = 0.
  for (y_true, y_pred) in zip(y_true_gram, y_pred_gram):
    batch_size = y_pred.shape[0]
    # WRONG!!! style_loss += tf.reduce_sum(tf.keras.losses.MSE(y_true[:batch_size], y_pred))/batch_size
    style_loss += tf.reduce_mean(MSELoss(y_true[:batch_size], y_pred))
  return style_loss  