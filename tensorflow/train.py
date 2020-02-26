import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import os
import random
import time
import PIL.Image as Image

# !git clone https://github.com/mixuala/tf_utils.git
from tf_utils import helpers
from tf_utils.io import load, show, save

# !git clone https://github.com/mixuala/fast_neural_style_pytorch.git
from fast_neural_style_pytorch.tensorflow import transformer, vgg, utils
import fast_neural_style_pytorch.utils as torch_utils

# paths
DATASET_PATH = "/content/train"
RECORD_PATH = '/content/coco2014-sq-1000.tfrecord'
VGG16_WEIGHTS_TORCH = '/content/vgg16-397923af.pth'
SAVE_MODEL_PATH = "/content/"
SAVE_IMAGE_PATH = "/content/"
STYLE_IMAGE_PATH = "/content/mosaic.jpg"

# !wget -O $STYLE_IMAGE_PATH https://raw.githubusercontent.com/iamRusty/fast-neural-style-pytorch/master/images/mosaic.jpg
# !wget -O $RECORD_PATH https://query.data.world/s/e7ny5zjv7rdklfhkg4uzaccsejm2s7



# GLOBAL SETTINGS
TRAIN_IMAGE_SIZE = 256
NUM_EPOCHS = 20
NUM_BATCHES = 250
BATCH_SIZE = 4 
CONTENT_WEIGHT = 17
STYLE_WEIGHT = 50
TV_WEIGHT = 1e-6 
ADAM_LR = 0.001
SAVE_MODEL_EVERY = 500 # 2,000 Images with batch size 4
SEED = 36
PLOT_LOSS = 1


xx_Dataset255 = utils.loadDataset(RECORD_PATH)
stacker = helpers.ImgStacker()
stacker.clear()
dbg = {}

def train():
  # # Seeds
  np.random.seed(SEED)
  tf.random.set_seed(SEED)

  # # Load networks
  TransformerNetwork = transformer.TransformerNetwork()
  style_model = vgg.get_layers("vgg16_torch")
  VGG = vgg.vgg_layers16( style_model['content_layers'], style_model['style_layers'], input_shape=(256,256,3) )

  # style_image can be any size, is static so we can cache features
  VGG_Target = vgg.vgg_layers16( style_model['content_layers'], style_model['style_layers'], input_shape=None )
  image_string = tf.io.read_file(STYLE_IMAGE_PATH)
  style_image = utils.ImageRecordDatasetFactory.image2tensor(image_string)*255.
  # style_image = utils.ImageRecordDatasetFactory.random_sq_crop(style_image, size=256)*255.
  style_tensor = tf.repeat( style_image[tf.newaxis,...], repeats=BATCH_SIZE, axis=0)
  show([style_image], w=128, domain=(0.,255.) )
  # B, H, W, C = style_tensor.shape
  (_, style_features) = VGG_Target( style_tensor , preprocess=True ) # hwcRGB
  target_style_gram = [ utils.gram(value)  for value in style_features ]  # list


  dbg['model'] = TransformerNetwork
  dbg['vgg'] = VGG
  dbg['target_style_gram'] = target_style_gram


  # # Optimizer settings
  optimizer = tf.optimizers.Adam(learning_rate=ADAM_LR, beta_1=0.99, epsilon=1e-1)

  # # Loss trackers
  content_loss_history = []
  style_loss_history = []
  total_loss_history = []
  batch_content_loss_sum = 0.
  batch_style_loss_sum = 0.
  batch_total_loss_sum = 0.

  # # Optimization/Training Loop
  @tf.function()
  def train_step(x_train, y_true, target_style_gram, loss_weights=None, log_freq=10):
    with tf.GradientTape() as tape:
      # Generate images and get features
      generated_batch = TransformerNetwork(x_train) # x_train domain=(0,255), hwcRGB

      # # apply scaled tanh
      # generated_batch = (tf.nn.tanh(generated_batch)+0.5)*255.

      hi = tf.reduce_max(generated_batch)
      tf.Assert( tf.greater(hi, 1.0),['RGB values should NOT be normalized'])
      # expecting generated_batch output domain=(0.,255.) for preprocess_input()
      (generated_content_features, generated_style_features) = VGG( generated_batch, preprocess=True )
      generated_style_gram = [ utils.gram(value)  for value in generated_style_features ]  # list
      dbg['generated_style_gram'] = generated_style_gram


      if y_true is None:
        y_true = x_train
        assert False, "y_true should not be None with tf.GradientTape"

      if tf.is_tensor(y_true):
        x_train = y_true # input domain=(0,255)
        # x_train_BGR_centered = tf.keras.applications.vgg16.preprocess_input(x_train*1.)/1.
        # x_train = tf.subtract(x_train, rgb_mean)
        target_content_features, _ = VGG(x_train, preprocess=True ) # hwcRGB

      elif isinstance(y_true, tuple):
        print("detect y_true is tuple(target_content_features + self.target_style_gram)", y_true[0].shape)
        target_content_features = y_true[:len(generated_content_features)]

      else:
        assert False, "unexpected result for y_true"


      # Content Loss
      # MSELoss = tf.keras.losses.MSE
      content_loss = utils.get_content_loss(target_content_features, generated_content_features)
      content_loss *= CONTENT_WEIGHT


      # # Style Loss
      style_loss = utils.get_style_loss(target_style_gram, generated_style_gram)
      style_loss *= STYLE_WEIGHT

      # Total Loss
      total_loss = content_loss + style_loss

      # apply batch_losses to grads
      grads = tape.gradient(total_loss, TransformerNetwork.trainable_weights)
      optimizer.apply_gradients(zip(grads, TransformerNetwork.trainable_weights))
      return (content_loss, style_loss, generated_batch)
      # end: with tf.GradientTape()
      # end: train_step()


  batch_count = 1
  start_time = time.time()

  for epoch in range(NUM_EPOCHS):

    train_dataset = xx_Dataset255.take(BATCH_SIZE * NUM_BATCHES).batch(BATCH_SIZE)

    print("========Epoch {}/{}========".format(epoch+1, NUM_EPOCHS))
    for x_train, y_true in train_dataset:
      # print("{}: batch shape={}".format(batch_count, x_train.shape))
      # Get current batch size in case of odd batch sizes
      curr_batch_size = x_train.shape[0]

      #   # Backprop and Weight Update
      (content_loss, style_loss, generated_batch) = train_step(x_train, y_true, target_style_gram) # 255.
      # print(">>> ",content_loss, style_loss)

      batch_content_loss_sum += content_loss
      batch_style_loss_sum += style_loss
      batch_total_loss_sum += (content_loss + style_loss)

      # Save Model and Print Losses
      if (((batch_count-1)%SAVE_MODEL_EVERY == 0) or (batch_count==NUM_EPOCHS*NUM_BATCHES)):

        # check side by side
        [rgb_image0, rgb_image] = [tf.squeeze(tf.clip_by_value(v[0], 0., 255.)) for v in [x_train, generated_batch]]
        # show([rgb_image0, rgb_image], domain=(0.,255.), w=128)
        check = stacker.hstack( rgb_image, limit=NUM_EPOCHS, smaller=True )
        show( check, domain=None )

        # Print Losses
        print("========Iteration {}/{}========".format(batch_count, NUM_EPOCHS*NUM_BATCHES))
        print("\tContent Loss:\t{:.2f}".format(batch_content_loss_sum/batch_count))
        print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum/batch_count))
        print("\tTotal Loss:\t{:.2f}".format(batch_total_loss_sum/batch_count))
        print("Time elapsed:\t{} seconds".format(time.time()-start_time))

        rgb_image = tf.squeeze(generated_batch[0,...]).numpy() # 255.
        check = np.stack([np.amin(rgb_image, axis=(0,1)), np.average(rgb_image, axis=(0,1)), np.amax(rgb_image, axis=(0,1))], axis=1)
        print( "\tImage, (min,mean,max):\t{}".format( check ) )

        # rgb_batch = tf.image.convert_image_dtype(generated_batch/255., dtype=tf.uint8, saturate=True)
        generated_batch = tf.clip_by_value(generated_batch, 0.,255.)
        show(generated_batch, w=128, domain=(0,255))

        # Save loss histories
        content_loss_history.append((batch_total_loss_sum/batch_count).numpy())
        style_loss_history.append((batch_style_loss_sum/batch_count).numpy())
        total_loss_history.append((batch_total_loss_sum/batch_count).numpy())

        # Save Model
        checkpoint_path = SAVE_MODEL_PATH + "tf_checkpoint_" + str(batch_count-1) + ".h5"
        TransformerNetwork.save_weights(checkpoint_path, save_format='h5')
        print("Saved tf TransformerNetwork checkpoint file at {}".format(checkpoint_path))

        # Save generated image
        sample_image_path = SAVE_IMAGE_PATH + "tf_sample_" + str(batch_count-1) + ".png"
        im=Image.fromarray(np.uint8(stacker.hstack()))
        im.save(sample_image_path)
        print("Saved sample tranformed image at {}".format(sample_image_path))

      # Iterate Batch Counter
      batch_count+=1

  stop_time = time.time()

  # Print loss histories
  print("Done Training the Transformer Network!")
  print("Training Time: {} seconds".format(stop_time-start_time))
  print("========Content Loss========")
  print(content_loss_history) 
  print("========Style Loss========")
  print(style_loss_history) 
  print("========Total Loss========")
  print(total_loss_history) 
  # Plot Loss Histories
  if (PLOT_LOSS):
      torch_utils.plot_loss_hist(content_loss_history, style_loss_history, total_loss_history)

train()