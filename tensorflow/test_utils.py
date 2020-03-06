import tensorflow as tf
import numpy as np
import PIL.Image


from fast_neural_style_pytorch.tensorflow import utils as fnstf_utils
from fast_neural_style_pytorch.tensorflow import vgg as tf_vgg
from fast_neural_style_pytorch.tensorflow import transformer as tf_transformer





class TransformerNetwork_VGG_ONES(tf.keras.Model):
  """TransformerNetwork_VGG that returns outputs as tf.ones()"""
  def __init__(self, style_image):
    super(TransformerNetwork_VGG_ONES, self).__init__()

    TransformerNetwork = tf_transformer.TransformerNetwork()
    TransformerNetwork.trainable = True
    style_model = tf_vgg.get_layers("vgg19")
    VGG = tf_vgg.vgg_layers19( style_model['content_layers'], style_model['style_layers'] )

    target_style_gram = TransformerNetwork_VGG_ONES._get_target_style_gram_from_image(style_image, style_model)
    nonzero = [tf.math.add(v,1.0) for v in target_style_gram]
    ones = [tf.math.divide_no_nan(v,v) for v in nonzero]
    vGGfeatures = VGG_Features(VGG, target_style_gram=tuple(ones))
    
    self.transformer = TransformerNetwork
    self.vgg = vGGfeatures

  def call(self, inputs):
    x = inputs
    x = self.transformer(x)
    # output_shapes = [(BATCH_SIZZE, 16, 16, 512),  (BATCH_SIZZE, 64, 64), (BATCH_SIZZE, 128, 128), (BATCH_SIZZE, 256, 256), (BATCH_SIZZE, 512, 512), (BATCH_SIZZE, 512, 512)]
    features = self.vgg(x)
    nonzero = [tf.math.add(v,1.0) for v in features]
    ones = [tf.math.divide_no_nan(v,v) for v in nonzero]
    return tuple(ones)

  @staticmethod
  def _get_target_style_gram_from_image(style_image, style_model):
    """"use when style_image.shape != (256,256,3)"""
    VGG_Target = tf_vgg.vgg_layers19( style_model['content_layers'], style_model['style_layers'], input_shape=None )
    if isinstance(style_image,str):
      image_string = tf.io.read_file(style_image)
      style_image = fnstf_utils.ImageRecordDatasetFactory.image2tensor(image_string, normalize=False)
    target_style_gram = VGG_Features.get_style_gram(VGG_Target, style_image)
    show([style_image], labels=["style_image, shape={}".format(style_image.shape)], w=128, domain=(0.,255.) )
    return target_style_gram



# ### unit tests
class UNIT_TEST():
  # static 
  TransformerNetwork_VGG_ONES = TransformerNetwork_VGG_ONES

  @staticmethod
  def inspect_vgg_features(features):
    """ features: features = VGG(generated_batch)"""
    assert isinstance(features, (tuple, dict)), "expecting tuple or dict got {}".format(type(features))
    if isinstance(features, tuple):
      for i, k in enumerate(features):
        if tf.is_tensor(k): print(i," >", k.shape)
        elif isinstance(k, (tuple, list)):
          _ = [ print(i,j," >>", v.shape) for j,v in enumerate(k)]
    else:
      for k in features:
        if tf.is_tensor(features[k]): print(k," >", features[k].shape)
        elif isinstance(features[k], (tuple, list)):
          _ = [ print(k," >>", v.shape) for v in features[k]]    

  @staticmethod
  def inpsect_model_losses(y_true, y_pred):
    """ check loss caclulations and loss weights"""
    print("y_pred: ")
    UNIT_TEST.inspect_vgg_features(y_pred)
    print("y_true: ")
    UNIT_TEST.inspect_vgg_features(y_true)

    if isinstance(y_pred, (tuple, list)):
      check1 = _content_loss_WEIGHTED(y_true[0], y_pred[0])
      check2 = _style_loss_WEIGHTED(y_true[:1], y_pred[:1])
      print("weighted losses", check1.numpy(), check2.numpy())

      check3 = utils.get_SUM_mse_loss(y_true[0], y_pred[0])
      check4 = utils.get_SUM_mse_loss(y_true[:1], y_pred[:1])
      print( "weights: ", CONTENT_WEIGHT, STYLE_WEIGHT)
      print("losses * weights", check3.numpy()*CONTENT_WEIGHT, check4.numpy()*STYLE_WEIGHT)


    if isinstance(y_pred, dict):
      check1 = _content_loss_WEIGHTED(y_true['content'], y_pred['content'])
      check2 = _style_loss_WEIGHTED(y_true['style'], y_pred['style'])
      print("weighted losses", check1.numpy(), check2.numpy())

      check3 = utils.get_SUM_mse_loss(y_true['content'], y_pred['content'])
      check4 = utils.get_SUM_mse_loss(y_true['style'], y_pred['style'])
      print( "weights: ", CONTENT_WEIGHT, STYLE_WEIGHT)
      print("losses * weights", check3.numpy()*CONTENT_WEIGHT, check4.numpy()*STYLE_WEIGHT)

    assert check2==check4*STYLE_WEIGHT, "style losses failed"
    assert check1==check3*CONTENT_WEIGHT, "content losses failed"


    @staticmethod
    def batch_generator_with_model_losses( transformer_vgg , BATCH_xy_Dataset255_with_TWOS):
      print("BATCH_dataset")

      # for x,y,weights in BATCH_xy_Dataset255_with_features.take(1):
      for x,y,weights in BATCH_xy_Dataset255_with_TWOS.take(1):

        print("FEATURE_WEIGHTS=", [tf.squeeze(v).numpy() for v in weights])
        # show(x, domain=None, w=128)
        # generated = TransformerNetwork(x)
        # show(generated, domain=None, w=128)

        features =  transformer_vgg(x)    

        # print("features")
        # UNIT_TEST_inspect_vgg_features(features)
        # print("y_true")
        # UNIT_TEST_inspect_vgg_features(y)

        if isinstance(features, (tuple, list)):

          print("\nFEATURE_WEIGHTS:",[v[0].numpy() for v in weights])
          print()

          check7 = utils.get_MEAN_mse_loss(y[0], features[0], weights[0]) 
          check8 = utils.get_MEAN_mse_loss(y[1:], features[1:], weights[1:])
          print("get_MEAN_mse_loss with FEATURE_WEIGHTS", check7.numpy(), check8.numpy())
          print()
          
          check5 = utils.get_SUM_mse_loss(y[0], features[0], weights[0]) 
          check6 = utils.get_SUM_mse_loss(y[1:], features[1:], weights[1:])
          print("get_SUM_mse_loss with FEATURE_WEIGHTS", check5.numpy(), check6.numpy())
          print()



        else:
          assert False, "features is probably a dict"


  @staticmethod
  def BATCH_xyGenerator_y_true_as_TWOS_and_weights(tensor_ds_255, 
                                                     VGGfeatures,
                                                     feature_weights=None
                                                     ):
    """ returns generator with weights(x_train, y_true, weights)
    """

    if feature_weights is not None: print("dataset generator using FEATURE_WEIGHTS=", feature_weights)
    
    def gen():
      weights = tuple( [v] for v in feature_weights) if feature_weights else tuple( [1.] for v in range(6))
      for x_train in tensor_ds_255.batch(BATCH_SIZE):
        batch = x_train if len(x_train.shape)==4 else x_train[tf.newaxis,...]

        # # # return as tuple( tensor, tuple) or tuple( tensor, ...)
        y_true_features = VGGfeatures(batch)
        if isinstance(y_true_features, (tuple, list)):
          if len(y_true_features)==2:
            # must FLATTEN to tuple( tensor, ...)
            content, style = y_true_features
            y_true_features = tuple([content] + list(style))
          else:
            pass # FLAT tuple(tensor x6) OK

        # ones
        nonzero = [tf.math.add(v,1.0) for v in y_true_features]
        ones = [tf.math.divide_no_nan(v,v) for v in nonzero]
        twos = [ v*2. for v in ones ]
        yield (x_train, tuple(twos), weights) 

    output_types= (
        tf.float32,    
        ( 
          tf.float32,       tf.float32, tf.float32, tf.float32, tf.float32, tf.float32
        ),
        (tf.float32,       tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
    )

    output_shapes = ( 
        (None, 256,256,3),
        (
          (None, 16, 16, 512),
          (None, 64, 64), (None, 128, 128), (None, 256, 256), (None, 512, 512), (None, 512, 512)
        ),
        (
            (1,),     (1,),(1,),(1,),(1,),(1,)
        )
    )
    return tf.data.Dataset.from_generator(
                                        generator=gen, 
                                        output_types=output_types, 
                                        output_shapes=output_shapes,
                                      )


  @staticmethod
  def check_mean_loss():
    # [OK] get_MEAN_mse_loss() gives correct loss value of FEATURE_WEIGHTS
    ONES = [1.,1.,1.,1.,1.,1.,]
    SEQ = [1.,2.,3.,4.,5.,6.,]
    WEIGHTS = SEQ
    transformerNetwork_VGG_ONES = UNIT_TEST.TransformerNetwork_VGG_ONES(style_image)
    BATCH_xy_Dataset255_with_TWOS = UNIT_TEST.BATCH_xyGenerator_y_true_as_TWOS_and_weights(
        tensor_ds_255, 
        transformerNetwork_VGG_ONES.vgg,
        feature_weights=WEIGHTS)
    train_dataset = BATCH_xy_Dataset255_with_TWOS.take(BATCH_SIZE * NUM_BATCHES)
    # force loss = 1.    
    for x,y,w in train_dataset.take(1):
      y_pred = transformerNetwork_VGG_ONES(x)
      loss = [ utils.get_MEAN_mse_loss(a,b, WEIGHTS[i]).numpy() for i,(a,b) in enumerate(zip(y,y_pred)) ] 
      # _ = [print( v[0,14:15,14, ...].numpy()) for v in y_pred]
      print("loss=", loss )
      assert loss==WEIGHTS    


  @staticmethod
  def check_multiple_output_loss_handling():
    SEQ = [1.,2.,3.,4.,5.,6.,]
    FEATURE_WEIGHTS = SEQ
    TransformerNetwork_VGG = UNIT_TEST.TransformerNetwork_VGG_ONES(style_image)
    BATCH_xy_Dataset255_with_features = UNIT_TEST.BATCH_xyGenerator_y_true_as_TWOS_and_weights(
          tensor_ds_255, 
          TransformerNetwork_VGG.vgg,
          feature_weights=FEATURE_WEIGHTS
        )
    train_dataset = BATCH_xy_Dataset255_with_features.take(BATCH_SIZE * NUM_BATCHES)
    # for x,y,w in BATCH_xy_Dataset255_with_features.take(1):
    #   print("check", [v[0].numpy() for v in w])


    def get_MEAN_mse_loss_TEST(y_true, y_pred):
      # CONFIRMED, TESTED OK
      # generator returns (x,y,w)
      # losses fed individually, without weights. 
      #     loss = [loss(x,y)*w, for x,y,w in zip(y_true, y_pred, weights)]
      assert not isinstance(y_pred, (tuple, list)), "expecting a tensor "
      return utils.get_MEAN_mse_loss(y_true, y_pred)


    TransformerNetwork_VGG.compile(
      optimizer=optimizer,
      loss=get_MEAN_mse_loss_TEST,
    )
    history = TransformerNetwork_VGG.fit(
      x=train_dataset.repeat(NUM_EPOCHS),
      epochs=NUM_EPOCHS,
      steps_per_epoch=NUM_BATCHES,
      callbacks=callbacks,  # NOT working
    )


