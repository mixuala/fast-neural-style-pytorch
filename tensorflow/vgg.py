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
  return VGG_Model(content_layers, style_layers, base_model, preprocessingFn=preprocess_input)
  
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
  return VGG_Model(content_layers, style_layers, base_model, preprocessingFn=preprocess_input)


class VGG_Model(tf.keras.models.Model):
  """ creates a VGG model that returns output values for the given layers
  see: https://keras.io/applications/#extract-features-from-an-arbitrary-intermediate-layer-with-vgg19

  Returns: 
    keras.models.Model(x, preprocess=True):
      Args: 
        x: image tuple/ndarray h,w,c(RGB), domain=(0.,255.)
      Returns:
        a tuple of lists, ([content_features], [style_features])

  usage:
    (content_features, style_features) = vgg_layers16(content_layers, style_layers, base_model, preprocess_input)(x_train)
  """  
  def __init__(self, content_layers, style_layers, base_model, preprocessingFn=None, **kwargs):
    super(VGG_Model, self).__init__(**kwargs)
    self.layer_names = content_layers + style_layers
    self.content_layer_split = len(content_layers)
    self.preprocessingFn = preprocessingFn
    base_model.trainable = False
    content_features = [base_model.get_layer(name).output for name in content_layers]
    style_features = [base_model.get_layer(name).output for name in style_layers]
    output_features = content_features + style_features
    self.model = Model( inputs=base_model.input, outputs=output_features, name="vgg_layers")
    self.model.trainable = False

  def call(self, x, preprocess=True):
    if preprocess and callable(self.preprocessingFn): 
      x = self.preprocessingFn(x)
    output = self.model(x) # call as tf.keras.Layer()
    return ( output[:self.content_layer_split], output[self.content_layer_split:] )




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


class VGG_Features():
  def __init__(self, loss_model, batch_size=4, style_image=None, target_style_gram=None):
    self.loss_model = loss_model
    self.batch_size = batch_size
    if style_image is not None:
      assert style_image.shape == (256,256,3), "ERROR: loss_model expecting input_shape=(256,256,3), got {}".format(style_image.shape)
      self.style_image = style_image
      self.target_style_gram = self.get_style_gram(self.style_image)
    if target_style_gram is not None:
      self.target_style_gram = target_style_gram

  def repeat_target_style_gram(self, batch_size):
    self.batch_size = batch_size
    self.target_style_gram = [ tf.repeat( gram[:1], repeats=batch_size, axis=0) for gram in self.target_style_gram]

  @staticmethod
  def get_style_gram(vggFeaturesModel, style_image, batch_size=4):
    style_batch = tf.repeat( style_image[tf.newaxis,...], repeats=batch_size, axis=0)
    # show([style_image], w=128, domain=(0.,255.) )

    # B, H, W, C = style_batch.shape
    (_, style_features) = vggFeaturesModel( style_batch , preprocess=True ) # hwcRGB
    target_style_gram = [ utils.gram(value)  for value in style_features ]  # list
    return target_style_gram  

  # UNUSED
  def get_vgg_losses(self, generated_batch, content_batch):
    generated_content_features, generated_style_features = self.loss_model( generated_batch, preprocess=True )
    generated_style_gram = [ utils.gram(value)  for value in generated_style_features ]  # list

    # Content Loss
    target_content_features, _ = self.loss_model(content_batch, preprocess=True )
    content_loss = utils.get_content_loss(target_content_features, generated_content_features)

    # # Style Loss
    style_loss = utils.get_style_loss(self.target_style_gram, generated_style_gram)
    return (content_loss, style_loss)

  def __call__(self, input_batch):
    content_features, style_features = self.loss_model( input_batch, preprocess=True )
    style_gram = [ utils.gram(value)  for value in style_features ]  # list
    # return (content_features[0], tuple(style_gram) )            # tuple of (tensor, tuple ) [OK], still counts as 6 outputs!!!
        
    # # name tensors for loss reporting
    content_f = tf.identity(content_features[0], name="content_0")
    style_f = [tf.identity(f, name="style_{}".format(i)) for i, f in enumerate(style_gram)]

    # # return as tuple 
    # return (content_f, tuple(style_f) )            # tuple of (tensor, tuple ) [OK], still counts as 6 outputs!!!

    # return as FLAT tuple 
    return tuple([content_f] + style_f )            # tuple of (tensor,... ) [OK], 6 outputs


  def get_dataset(self, tensor_ds, batch_size=None, feature_weights=None):
    """ get a dataset.from_generator() that works with the feature outputs, e.g. VGG_Features(...)(x_train)

    usage: 
      train_ds = VGG_Features(...).get_dataset( tensor_ds, batch_size=BATCH_SIZE )
      train_ds = train_ds.take(BATCH_SIZE * NUM_BATCHES).repeat(NUM_EPOCHS)

    Args:
      tensor_ds: a dataset of (256,256,3), domain=(0.,255.) image tensors

    Returns:
      a dataset from generator
    """
    if batch_size is not None:
      if self.batch_size != batch_size:
        self.repeat_target_style_gram(batch_size)
        

      # get VGG_Features per batch, more efficient
      def _gen(tensor_ds_255):

        weights = tuple( [v] for v in feature_weights) if feature_weights else tuple( [1.] for v in range(6))

        for x_train in tensor_ds_255.batch(batch_size):
          batch = x_train if len(x_train.shape)==4 else x_train[tf.newaxis,...]

          assert tf.reduce_max(x_train)>1.0, "ERROR: expecting image domain=(0.,255.)"
          assert len(x_train)==4, "ERROR: expecting tensor_ds to return unbatched image tensors"

          # # # return as tuple( tensor, tuple) or tuple( tensor, ...)
          y_true_features = self.__call__(x_train)
          if isinstance(y_true_features, (tuple, list)):
            if len(y_true_features)==2:
              # must FLATTEN to tuple( tensor, ...)
              content, style = y_true_features
              y_true_features = tuple([content] + list(style))
            else:
              pass # FLAT tuple(tensor x6) OK

          elif isinstance(y_true_features, dict): 
            # return as dict, see VGGfeatures.__call__()
            # y_true_features = VGGfeatures(x_train)
            pass

          yield (x_train, y_true_features, weights)
          

      generator = lambda tensor_ds: _gen(tensor_ds)
      # ERROR: do NOT return as tuple(tensor, tuple), ERROR: AttributeError: 'tuple' object has no attribute 'shape'
      # # as FLAT tuple      
      output_types=(
          tf.float32,     # x_train
          (               # y_true
            tf.float32,               tf.float32, tf.float32, tf.float32, tf.float32, tf.float32
          ),
          (tf.float32,       tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
      )
      output_shapes = ( 
          (None, 256,256,3),
          (
            (None, 16, 16, 512),      (None, 64, 64), (None, 128, 128), (None, 256, 256), (None, 512, 512), (None, 512, 512)
          ),
          (
              (1,),     (1,),(1,),(1,),(1,),(1,)
          )
      )
    elif "return as dict" and False:
      def _gen(tensor_ds_255):
        for batch in tensor_ds_255.batch(batch_size):
          content, style = self.__call__(batch) 
          y_true_features = {
            'content': tuple( content ), 
            'style':  tuple( style )
          }
          yield (x_train, y_true_features)

      generator = lambda tensor_ds: _gen(tensor_ds)
      output_types=( tf.float32, {
          "content": (tf.float32), 
          "style": (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
      })
      output_shapes = ( 
          (None, 256,256,3), {
            "content": (None, 16, 16, 512),
            "style":(
              (None, 64, 64), (None, 128, 128), (None, 256, 256), (None, 512, 512), (None, 512, 512)
            )
      })

    else:
      # get VGG_Features for single image
      def _gen(tensor_ds_255):
        for x_train in tensor_ds_255:
          batch = x_train if len(x_train.shape)==4 else x_train[tf.newaxis,...]
          content, style = self.__call__(batch) 
          y_true_features = (tf.squeeze(content[:1]), tuple([tf.squeeze(v) for v in style]))  # tuple( 6 feature tensors )
          yield (x_train, y_true_features)

      generator = lambda tensor_ds: _gen(tensor_ds)
      output_types=(
          tf.float32,     # x_train
          (               # y_true
            (tf.float32),    
            (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)  
          )
      ),
      output_shapes = ( 
          (256,256,3),
          (
            ( (16, 16, 512) ),
            ( (64, 64), (128, 128), (256, 256), (512, 512), (512, 512) )
          )
      )

    return tf.data.Dataset.from_generator(
                                      generator=generator, 
                                      output_types=output_types, 
                                      output_shapes=output_shapes,
                                    )


