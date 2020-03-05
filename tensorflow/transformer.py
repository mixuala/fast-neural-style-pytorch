import tensorflow as tf
import numpy as np


from fast_neural_style_pytorch.tensorflow import vgg
from fast_neural_style_pytorch.tensorflow import utils


class ReflectionPadding2D(tf.keras.layers.Layer):
  """
  usage: 
    reflect2D = ReflectionPadding2D(k)
    x = reflect2D(input)
  """
  def __init__(self, kernel_size):
    super(ReflectionPadding2D, self).__init__()
    pad = kernel_size//2
    # self.paddings = tf.constant([[pad, pad], [pad, pad]])
    self.paddings = tf.constant([ [0,0], [pad, pad], [pad, pad], [0,0] ])

  def call(self, input_tensor):
    return tf.pad( input_tensor, self.paddings, "REFLECT")

class InstanceNorm2D(tf.keras.layers.Layer):
  """use Ulyanov's instance normalization over batch norm

  uses CHANNELS_LAST

  see: https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py
  """  

  def __init__(self):
    super(InstanceNorm2D, self).__init__()
  
  # TODO: unit test against `nn.InstanceNorm2d(out_channels, affine=True)`
  def call(self, input_tensor, training=True):
    batch, rows, cols, channels = input_tensor.shape
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(input_tensor, [1,2], keepdims=True)
    epsilon = 1e-3
    normalized = (input_tensor-mu)/(sigma_sq + epsilon)**(.5)
    return normalized
    # shift = tf.Variable( lambda: tf.zeros(var_shape))
    # scale = tf.Variable( lambda: tf.ones(var_shape))
    # normalized = (input_tensor-mu)/(sigma_sq + epsilon)**(.5)
    # return scale * normalized + shift


class ConvLayer(tf.keras.layers.Layer):
  """
  reflection_padding before conv2d
  instance normalization
  activation="relu" or None
  """

  def __init__(self,  f=128, k=3, s=1, norm="instance", activation=None, **kwargs):
    name = kwargs.get('name', None)
    super(ConvLayer, self).__init__(name=name)
    self.reflection_padding = ReflectionPadding2D(k)
    self.conv2d = tf.keras.layers.Conv2D(filters=f, kernel_size=k, strides=s,
                                        padding="valid", activation=activation,
    )
    self.norm_type = norm
    if (norm=="instance"):
      self.norm_layer = InstanceNorm2D()
    elif (norm=="batch"):
      self.norm_layer = tf.keras.layers.BatchNormalization()
    else:
      self.norm_layer = None

  def call(self, input_tensor, training=True):
    """forward pass"""
    x = input_tensor
    x = self.reflection_padding(x)
    x = self.conv2d(x, training=training)
    if self.norm_layer is not None:
      x = self.norm_layer(x, training=training)

    return x


class ResidualLayer(tf.keras.layers.Layer):
  """
  Deep Residual Learning for Image Recognition

  usage: res_block = ResidualLayer(128,3,1)

  https://arxiv.org/abs/1512.03385

  from pytorch, see:
  - https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/transformer_net.py
  - https://github.com/rrmina/fast-neural-style-pytorch/blob/master/transformer.py
  """
  def __init__(self,  f=128, k=3, s=1, **kwargs):
    name = kwargs.get('name', None)
    norm = kwargs.get('norm', "instance")
    super(ResidualLayer, self).__init__(name=name)
    self.conv1 = ConvLayer(f,k,s, norm=norm, activation=None)
    self.conv2 = ConvLayer(f,k,s, norm=norm, activation=None)

  def call(self, input_tensor, training=True):
    """forward pass"""
    x = input_tensor
    x = self.conv1(x)
    x = tf.nn.relu(x)
    x = self.conv2(x)
    x = tf.add(x, input_tensor, name="residual")
    return x

class DeconvLayer(tf.keras.layers.Layer):
  """UpsampleConvLayer
  
  Upsamples the input and then does a convolution. This method gives better results
  compared to ConvTranspose2d.
  ref: http://distill.pub/2016/deconv-checkerboard/
  """
  def __init__(self,  f=128, k=3, s=1, norm="instance", mode="upsample", activation=None, **kwargs):
    name = kwargs.get('name', None)
    norm = kwargs.get('norm', None)
    super(DeconvLayer, self).__init__(name=name)

    # ???: what is the correct padding for Conv2DTranspose?
    # self.conv_transpose = tf.keras.layers.Conv2DTranspose(filters=f, kernel_size=k, strides=s,
    #                                       padding="same", activation=activation,
    # )
    assert mode=="upsample", "ERROR: Conv2DTranspose not implemented"
    self.mode = mode
    self.upsample = tf.keras.layers.UpSampling2D(2)
    self.conv2d = ConvLayer(f,k,s, norm=norm, activation=activation)

    self.norm_type = norm
    if (norm=="instance"):
      self.norm_layer = InstanceNorm2D()
    elif (norm=="batch"):
      self.norm_layer = tf.keras.layers.BatchNormalization()
    else:
      self.norm_layer = None

  def call(self, input_tensor, training=True):
    """forward pass"""
    x = input_tensor
    x = self.upsample(x)
    x = self.conv2d(x, training=training)
    if self.norm_layer is not None:
      x = self.norm_layer(x, training=training)
    return x  



class TransformerNetwork(tf.keras.Model):
  """same as TransformerNetwork, but avoid using `tf.keras.Sequential()` for 
  better results with `tf.keras.utils.plot_model()`
  """
  def __init__(self, norm="instance", **kwargs):
    name = kwargs.get('name', None)
    super(TransformerNetwork, self).__init__(name=name)
    self.norm = norm

    self.conv1 = ConvLayer( 32,9,1, name="conv_1", norm=self.norm, activation=None)
    self.conv2 = ConvLayer( 64,3,2, name="conv_2", norm=self.norm, activation=None)
    self.conv3 = ConvLayer(128,3,2, name="conv_3", norm=self.norm, activation=None)

    self.res1 = ResidualLayer(128, 3, name="residual_1", norm=self.norm)
    self.res2 = ResidualLayer(128, 3, name="residual_2", norm=self.norm)
    self.res3 = ResidualLayer(128, 3, name="residual_3", norm=self.norm)
    self.res4 = ResidualLayer(128, 3, name="residual_4", norm=self.norm)
    self.res5 = ResidualLayer(128, 3, name="residual_5", norm=self.norm)

    self.deconv1 = DeconvLayer( 64,3,1, name="deconv_1", norm=self.norm, activation=None)
    self.deconv2 = DeconvLayer( 32,3,1, name="deconv_2", norm=self.norm, activation=None)
    self.deconv3 = ConvLayer(3, 9, 1, name="deconv_3", norm=None, activation=None)


  def call(self, input_tensor, training=True):
    """forward pass"""    
    ### feed_forward one input and get learned learned_image/output from layers
    x = input_tensor
    if x.dtype != 'float32': 
      x = tf.image.convert_image_dtype(x, tf.float32)

    # x = self.ConvBlock(x)
    x = self.conv1(x)
    x = tf.nn.relu(x)
    x = self.conv2(x)
    x = tf.nn.relu(x)
    x = self.conv3(x)
    x = tf.nn.relu(x)

    # x = self.ResBlock(x)
    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.res4(x)
    x = self.res5(x)

    # x = self.DeconvBlock(x)
    x = self.deconv1(x)
    x = tf.nn.relu(x)
    x = self.deconv2(x)
    x = tf.nn.relu(x)
    x = self.deconv3(x)
    return x



class TransformerNetworkTanh(TransformerNetwork):
    """A modification of the transformation network that uses Tanh function as output 
    This follows more closely the architecture outlined in the original paper's supplementary material
    this model produces darker images and provides retro styling effect
    Reference: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf

    NOTE: outputs are NOT clipped to image RGB values
    """
    # override __init__ method
    def __init__(self, tanh_multiplier=150, **kwargs):
      super(TransformerNetworkTanh, self).__init__(**kwargs)
      self.tanh = tf.keras.activations.tanh
      self.multiplier = tanh_multiplier


    def call(self, input_tensor, training=True):
      """forward pass"""
      x = input_tensor
      x = super(TransformerNetworkTanh, self).call(x, training=training)
      x = self.tanh(x)
      x *=  self.multiplier
      return x





# check.build(input_shape=(None,255,255,3))
# check.summary()
# check = TransformerNetwork(name="transformer")
# print("***\n")
# tf.keras.utils.plot_model(check, 'transformer.png', expand_nested=True, show_shapes=True)



def getTransformerNetworkFn(shape=tuple, norm="instance", **kwargs):
  """same as TransformerNetwork, but use keras functional API for 
  better results with `tf.keras.utils.plot_model()`
  """
  from types import SimpleNamespace
  _layers = SimpleNamespace()
  _layers.name = kwargs.get('name', None)
  _layers.norm = norm
  _layers.conv1 = ConvLayer( 32,9,1, name="conv_1", norm=_layers.norm, activation=None)
  _layers.conv2 = ConvLayer( 64,3,2, name="conv_2", norm=_layers.norm, activation=None)
  _layers.conv3 = ConvLayer(128,3,2, name="conv_3", norm=_layers.norm, activation=None)
  _layers.res1 = ResidualLayer(128, 3, name="residual_1")
  _layers.res2 = ResidualLayer(128, 3, name="residual_2")
  _layers.res3 = ResidualLayer(128, 3, name="residual_3")
  _layers.res4 = ResidualLayer(128, 3, name="residual_4")
  _layers.res5 = ResidualLayer(128, 3, name="residual_5")
  _layers.deconv1 = DeconvLayer( 64,3,1, name="deconv_1", norm=_layers.norm, activation=None)
  _layers.deconv2 = DeconvLayer( 32,3,1, name="deconv_2", norm=_layers.norm, activation=None)
  _layers.deconv3 = ConvLayer(3, 9, 1, name="deconv_3", norm=None, activation=None)

  input_tensor = tf.keras.Input( shape=shape )
  
  ### feed_forward one input and get learned learned_image/output from layers
  x = input_tensor
  if x.dtype != 'float32': 
    x = tf.image.convert_image_dtype(x, tf.float32)

  # x = self.ConvBlock(x)
  x = _layers.conv1(x)
  x = tf.nn.relu(x)
  x = _layers.conv2(x)
  x = tf.nn.relu(x)
  x = _layers.conv3(x)
  x = tf.nn.relu(x)

  # x = _layers.ResBlock(x)
  x = _layers.res1(x)
  x = _layers.res2(x)
  x = _layers.res3(x)
  x = _layers.res4(x)
  x = _layers.res5(x)

  # x = _layers.DeconvBlock(x)
  x = _layers.deconv1(x)
  x = tf.nn.relu(x)
  x = _layers.deconv2(x)
  x = tf.nn.relu(x)
  x = _layers.deconv3(x)
  return tf.keras.Model(input_tensor, x)



class TransformerNetwork_VGG(tf.keras.Model):
  def __init__(self, style_image, **kwargs):
    super(TransformerNetwork_VGG, self).__init__(**kwargs)

    transformerNetwork = TransformerNetwork()
    transformerNetwork.trainable = True
    style_model = vgg.get_layers("vgg19")
    VGG = vgg.vgg_layers19( style_model['content_layers'], style_model['style_layers'] )

    if tf.is_tensor(style_image) and style_image.shape==(256,256,3):
      VGGfeatures = vgg.VGG_Features(VGG, style_image=style_image)
    else: 
      target_style_gram = TransformerNetwork_VGG._get_target_style_gram_from_image(style_image, style_model)
      VGGfeatures = vgg.VGG_Features(VGG, target_style_gram=target_style_gram)
    
    self.transformer = transformerNetwork
    self.vgg = VGGfeatures

  def call(self, inputs):
    x = inputs
    x = self.transformer(x)
    x = self.vgg(x)
    return x

  @staticmethod
  def _get_target_style_gram_from_image(style_image, style_model):
    """"use when style_image.shape != (256,256,3)"""
    VGG_Target = vgg.vgg_layers19( style_model['content_layers'], style_model['style_layers'], input_shape=None )
    if isinstance(style_image,str):
      image_string = tf.io.read_file(style_image)
      style_image = utils.ImageRecordDatasetFactory.image2tensor(image_string, normalize=False)
    target_style_gram = vgg.VGG_Features.get_style_gram(VGG_Target, style_image)
    show([style_image], labels=["style_image, shape={}".format(style_image.shape)], w=128, domain=(0.,255.) )
    return target_style_gram

