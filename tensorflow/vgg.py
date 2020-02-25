import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from fast_neural_style_pytorch.tensorflow import utils

def get_layers(model="vgg19"):
  layers = { 
    "vgg16_torch" : {}, 
    "vgg19" : {},
    "vgg19_torch" : {},
  }
  # Content layer where will pull our feature maps
  layers["vgg16_torch"]['content_layers'] = ['block2_conv2'] 
  layers["vgg16_torch"]['style_layers'] = ['block1_conv2',
                  'block2_conv2',
                  'block3_conv3', 
                  'block4_conv3']

  # Content layer where will pull our feature maps
  layers["vgg19"]['content_layers'] = ['block5_conv2'] 
  layers["vgg19"]['style_layers'] = ['block1_conv1',
                  'block2_conv1',
                  'block3_conv1', 
                  'block4_conv1', 
                  'block5_conv1']

  # mapped from layers used by pytorch: https://github.com/rrmina/fast-neural-style-pytorch
  layers["vgg19_torch"]['content_layers'] = ['block2_conv2']
  layers["vgg19_torch"]['style_layers'] = ['block1_conv2',
                'block2_conv2',
                'block3_conv4', 
                'block4_conv2', 
                'block4_conv4',
                'block5_conv4',
                ]
  return layers[model]

def vgg_layers16(content_layers, style_layers, input_shape=(256,256,3)):
  """ creates a VGG model that returns output values for the given layers
  see: https://keras.io/applications/#extract-features-from-an-arbitrary-intermediate-layer-with-vgg19

  Returns: 
    function(x, preprocess=True):
      Args: 
        x: image tuple/ndarray h,w,c(RGB), domain=(0.,255.)
      Returns:
        a tuple of lists, ([content_features], [style_features])

  usage:
    (content_features, style_features) = vgg_layers16(content_layers, style_layers)(x_train)
  """
  from tensorflow.keras.applications.vgg16 import preprocess_input
  base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
  return vgg_layersXX(content_layers, style_layers, base_model, preprocess_input)
  
def vgg_layers19(content_layers, style_layers, input_shape=(256,256,3)):
  """ creates a VGG model that returns output values for the given layers
  see: https://keras.io/applications/#extract-features-from-an-arbitrary-intermediate-layer-with-vgg19

  Returns: 
    function(x, preprocess=True):
      Args: 
        x: image tuple/ndarray h,w,c(RGB), domain=(0.,255.)
      Returns:
        a tuple of lists, ([content_features], [style_features])

  usage:
    (content_features, style_features) = vgg_layers16(content_layers, style_layers)(x_train)
  """
  from tensorflow.keras.applications.vgg19 import preprocess_input
  base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
  return vgg_layersXX(content_layers, style_layers, base_model, preprocess_input)
  

def vgg_layersXX(content_layers, style_layers, base_model, preprocessingFn=None):
  """ creates a VGG model that returns output values for the given layers
  see: https://keras.io/applications/#extract-features-from-an-arbitrary-intermediate-layer-with-vgg19

  Returns: 
    function(x, preprocess=True):
      Args: 
        x: image tuple/ndarray h,w,c(RGB), domain=(0.,255.)
      Returns:
        a tuple of lists, ([content_features], [style_features])

  usage:
    (content_features, style_features) = vgg_layers16(content_layers, style_layers, base_model, preprocess_input)(x_train)
  """
  base_model.trainable = False
  layer_names = content_layers + style_layers
  content_features = [base_model.get_layer(name).output for name in content_layers]
  style_features = [base_model.get_layer(name).output for name in style_layers]
  output_features = content_features + style_features

  model = Model( inputs=base_model.input, outputs=output_features, name="vgg_layers")
  model.trainable = False

  def _get_features(x, preprocess=True):
    """
    Args:
      x: expecting tensor, domain=255. hwcRGB
    """
    if preprocess and callable(preprocessingFn): 
      x = preprocessingFn(x)
    output = model(x) # call as tf.keras.Layer()
    return ( output[:len(content_layers)], output[len(content_layers):] )

  return _get_features  



class StyleContentModel(tf.keras.Model):
  """model that returns style and content tensors for input image using VGG19
  
  use to get output tensors for [content, style, transfer] images. cache [content, style] outputs


  usage:
    mode = layers["vgg19_torch"]
    StyleContentModel(mode.content_layers, mode.style_layers)
  """
  style_layers = ["layer_name"]
  content_layers = ["layer_name"]
  vgg = None # tf.keras.Model()

  def __init__(self, content_layers, style_layers, **kwargs):

    """model constructor

    define model layers in constructor
    """
    super(StyleContentModel, self).__init__(**kwargs)
    self.content_layers = content_layers
    self.style_layers = style_layers
    self.num_style_layers = len(style_layers)
    self.vgg = vgg_layers19(content_layers, style_layers, input_shape=(256,256,3) ) # tf.keras.Layer()
    self.trainable = False


  def call(self, inputs, preprocess=True):
    """model forward pass implementation

    Args:
      inputs: transfer_image, hwc(RGB), domain=(0.,255.)
    """
    rank = len(inputs.shape)
    if rank==3: inputs = inputs[tf.newaxis, ...]
    assert len(vgg_input.shape)==4, "expecting a batch of image, shape=(?,h,w,c), got={}".format(vgg_input.shape)

    content_outputs, style_outputs = self.vgg(vgg_input, preprocess=preprocess) # tf.keras.Model()
    # apply gram_matrix to style outputs
    style_outputs = [utils.gram(v) for v in style_outputs]
    return (content_outputs, style_outputs)  # list()
