
import utils

layers = { 
  # "vgg16" : {}, 
  "vgg19" : {},
  "vgg19_torch" : {},
}

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


def vgg_layers0(content_layers, style_layers):
  """ creates a VGG model that returns output values for the given layers   

  Returns: tuple of lists, (content_features, style_features)

  usage:
    (content_features, style_features) = vgg_layers(content_layers, style_layers)(x_train)
  """
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  layer_names = content_layers + style_layers
  content_features = [vgg.get_layer(name).output for name in content_layers]
  style_features = [vgg.get_layer(name).output for name in style_layers]
  output_features = content_features + style_features

  def _get_features(x, preprocess=True):
    # ???: preprocess vgg_input, BGR, mean_centered
    if preprocess:
      x = utils.vgg_input_preprocess(x)
    output = tf.keras.Model( inputs=vgg.input, outputs=output_features, name="vgg_layers")(x)
    return ( output[:len(content_layers)], output[len(content_layers):] )

  return _get_features
  


def vgg_layers(layer_names):
  """ creates a VGG model that returns output values for the given layers 
  Returns: tf.keras.Model()
  
  usage:
    y_pred = vgg_layers(layer_names)(x_train)
  """
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model( inputs=vgg.input, outputs=outputs, name="vgg_layers")
  return model


class StyleContentModel(tf.keras.Model):
  """model that returns style and content tensors for input image
  
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
    self.vgg = vgg_layers(content_layers + style_layers ) # tf.keras.Model()
    self.trainable = False


  def call(self, inputs, preprocess=True):
    """model forward pass implementation

    Args:
      inputs: transfer_image, domain=(0.,1.)
    """
    rank = len(inputs.shape)
    if rank==3: inputs = inputs[tf.newaxis, ...]

    if preprocess or True:
      # self.style_target = self.vgg_losses(bgr_inputs)[1:] # drop xc
      vgg_input = vgg_input_preprocess(inputs)

    else:
      vgg_input = inputs

    assert len(vgg_input.shape)==4, "expecting a batch of image, shape=(?,h,w,c), got={}".format(vgg_input.shape)

    vgg_outputs = self.vgg(vgg_input) # tf.keras.Model()
    split = len(self.content_layers)
    content_outputs, style_outputs = (vgg_outputs[:split], vgg_outputs[split:])
    # apply gram_matrix to style outputs
    style_outputs = [utils.gram(v) for v in style_outputs]
    return content_outputs + style_outputs  # list()
