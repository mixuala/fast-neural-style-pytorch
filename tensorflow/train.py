%tensorflow_version 2.x
import tensorflow as tf

import os
import random
import numpy as np
import time

import vgg
import transformer
import utils

# !rm -rf tf_utils
# !git clone https://github.com/mixuala/tf_utils.git
# from tf_utils import helpers
# from tf_utils.io import load, show, save

# GLOBAL SETTINGS
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "/content/train"
NUM_EPOCHS = 1
NUM_BATCHES = None
STYLE_IMAGE_PATH = "/content/mosaic.jpg"
BATCH_SIZE = 4 
CONTENT_WEIGHT = 17
STYLE_WEIGHT = 50
TV_WEIGHT = 1e-6 
ADAM_LR = 0.001
SAVE_MODEL_PATH = "/content/"
SAVE_IMAGE_PATH = "/content/"
# SAVE_MODEL_EVERY = 500 # 2,000 Images with batch size 4
SAVE_MODEL_EVERY = 10 # 2,000 Images with batch size 4
SEED = 35
PLOT_LOSS = 1

# # Dataset and Dataloader
def transforms(filename):
  """ resize SMALLEST dim to TRAIN_IMAGE_SIZE, then grab a square center crop
  """
  image, label = utils.sq_center_crop(filename, size=TRAIN_IMAGE_SIZE, normalize=True)
  return image, label

def torch_transforms(filename):
  """
  same as pytorch implementation, but returns CHANNELS_LAST tf.tensor()
  """
  import torchvision
  parts = tf.strings.split(filename, '/')
  label = parts[-2]
  pil_image = tf.keras.preprocessing.image.load_img(filename)
  pil_image = torchvision.transforms.Resize(TRAIN_IMAGE_SIZE)(pil_image)
  pil_image = torchvision.transforms.CenterCrop(TRAIN_IMAGE_SIZE)(pil_image)
  image = np.array( torchvision.transforms.ToTensor()(pil_image) ) # CHANNELS_FIRST
  image = tf.transpose(image, perm=[1,2,0]) # normalized, CHANNELS_LAST
  # image *= 255. # 255, CHANNELS_LAST 
  return image, label

def tf_transforms(filename):
  """
  works with Dataset.map(), but distorts image on resize
  """
  parts = tf.strings.split(filename, '/')
  label = parts[-2]
  
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image, channels=3) # 255
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE])
  return image, label  


def train():
  # # Seeds
  # torch.manual_seed(SEED)
  # torch.cuda.manual_seed(SEED)
  np.random.seed(SEED)
  tf.random.set_seed(SEED)
  random.seed(SEED)

  # # Dataset and Dataloader
  list_ds = tf.data.Dataset.list_files('{}/*/*'.format(DATASET_PATH))
  NUM_BATCHES = 100
  list_ds = list_ds.take(BATCH_SIZE * NUM_BATCHES)
  # train_dataset = list_ds.map(transforms).shuffle(buffer_size=100).batch(BATCH_SIZE)
  train_dataset = utils.batch_torch_transforms(list_ds.shuffle(buffer_size=100), BATCH_SIZE) # NOT lazy loaded

  # # Load networks
  TransformerNetwork = transformer.TransformerNetwork()
  style_model = vgg.get_layers("vgg19_torch")
  VGG = vgg.vgg_layers0( style_model['content_layers'], style_model['style_layers'] )

  # # Get Style Features, BGR ordering
  rgb_mean = tf.reshape(tf.constant( [123.68, 116.779, 103.939], dtype=tf.float32), (1,1,1,3))
  rgb_mean_normalized = tf.reshape(tf.constant( [0.48501961, 0.45795686, 0.40760392], dtype=tf.float32), (1,1,1,3))

  # style_image is static, cache features
  style_image = tf.keras.preprocessing.image.load_img(STYLE_IMAGE_PATH)
  style_image = tf.keras.preprocessing.image.img_to_array(style_image, dtype=float)/255.
  style_tensor = tf.repeat( style_image[tf.newaxis,...], repeats=BATCH_SIZE, axis=0)
  # B, H, W, C = style_tensor.shape
  # (_, style_features) = VGG( style_tensor , preprocess=True )
  style_tensor = tf.keras.applications.vgg16.preprocess_input(tf.cast(style_tensor, tf.float32)*255.)/255.
  (_, style_features) = VGG( style_tensor , preprocess=False )
  target_style_gram = [ utils.gram(value)  for value in style_features ]  # list

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
  def train_step(x_train, y_train=None, loss_weights=None, log_freq=10):
    with tf.GradientTape() as tape:
      # Generate images and get features
      # content_batch = content_batch[:,[2,1,0]].to(device) # 
      generated_batch = TransformerNetwork(x_train)
      # TODO: clip and/or tanh before VGG()??, what is the domain of generated_batch?
      generated_batch = tf.add(tf.nn.tanh(generated_batch), rgb_mean_normalized)
  
      # (generated_content_features, generated_style_features) = VGG( generated_batch, preprocess=True )
      generated_batch_BGR_centered = tf.keras.applications.vgg16.preprocess_input(generated_batch*255.)/255.
      (generated_content_features, generated_style_features) = VGG( generated_batch_BGR_centered, preprocess=False )

      generated_style_gram = [ utils.gram(value)  for value in generated_style_features ]  # list

      # (content_features, _) = VGG(x_train, preprocess=True )
      x_train = tf.keras.applications.vgg16.preprocess_input(tf.cast(x_train, tf.float32)*255.)/255.
      (content_features, _) = VGG(x_train, preprocess=False )

      # Content Loss
      MSELoss = tf.keras.losses.MSE
      content_loss = 0.
      for (y_true, y_pred) in zip(generated_content_features, content_features):
        content_loss += tf.reduce_sum(MSELoss(y_true, y_pred))
      content_loss *= CONTENT_WEIGHT      

      # Style Loss
      style_loss = 0.
      for (y_true, y_pred)in zip(target_style_gram, generated_style_gram):
        style_loss += MSELoss(y_true, y_pred)
      style_loss *= STYLE_WEIGHT
      batch_style_loss_sum += style_loss

      # Total Loss
      total_loss = content_loss + style_loss
      batch_total_loss_sum += total_loss  # ???: how do I use this with GradientTape?
      # end: with tf.GradientTape()

    # apply batch_losses to grads
    grads = tape.gradient(total_loss, TransformerNetwork.trainable_weights)
    optimizer.apply_gradients(zip(grads, TransformerNetwork.trainable_weights))
    return (content_loss, style_loss, generated_batch)
    # end: train_step()


  batch_count = 1
  start_time = time.time()
  for epoch in range(NUM_EPOCHS):

    if "use loss/current batch" and False:
      # ???: what do we want? loss/batch or *cumulative* loss/batch?
      batch_content_loss_sum = 0
      batch_style_loss_sum = 0
      batch_total_loss_sum = 0
      batch_count = 1

    print("========Epoch {}/{}========".format(epoch+1, NUM_EPOCHS))
    # for content_batch, _ in train_loader:
    for content_batch, label in train_dataset:
      # print("{}: batch shape={}".format(batch_count, content_batch.shape))
      # Get current batch size in case of odd batch sizes
      curr_batch_size = content_batch.shape[0]

      #   # Backprop and Weight Update
      (content_loss, style_loss, generated_batch) = train_step(content_batch)
      batch_content_loss_sum += content_loss
      batch_style_loss_sum += style_loss
      batch_total_loss_sum += (content_loss + style_loss)

      # Save Model and Print Losses
      if (((batch_count-1)%SAVE_MODEL_EVERY == 0) or (batch_count==NUM_EPOCHS*NUM_BATCHES)):
        # Print Losses
        print("========Iteration {}/{}========".format(batch_count, NUM_EPOCHS*NUM_BATCHES))
        print("\tContent Loss:\t{:.2f}".format(batch_content_loss_sum/batch_count))
        print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum/batch_count))
        print("\tTotal Loss:\t{:.2f}".format(batch_total_loss_sum/batch_count))
        print("Time elapsed:\t{} seconds".format(time.time()-start_time))

        # # Save Model
        # checkpoint_path = SAVE_MODEL_PATH + "checkpoint_" + str(batch_count-1) + ".pth"
        # torch.save(TransformerNetwork.state_dict(), checkpoint_path)
        # print("Saved TransformerNetwork checkpoint file at {}".format(checkpoint_path))

        # Save sample generated image
        check_img = generated_batch[0,...] # 255.
        print( "\tImage, domain:\t({:.1f},{:.1f})".format( tf.reduce_min(check_img).numpy(), tf.reduce_max(check_img).numpy() ) )
        rgb_img = tf.image.convert_image_dtype(check_img/255., tf.uint8)
        show(rgb_img)  # use `rgb_img` for display only, auto clip to 255

        # sample_image = utils.ttoi(sample_tensor.clone().detach())
        # sample_image_path = SAVE_IMAGE_PATH + "sample0_" + str(batch_count-1) + ".png"
        # utils.saveimg(sample_image, sample_image_path)
        # print("Saved sample tranformed image at {}".format(sample_image_path))

        # Save loss histories
        content_loss_history.append(batch_total_loss_sum/batch_count)
        style_loss_history.append(batch_style_loss_sum/batch_count)
        total_loss_history.append(batch_total_loss_sum/batch_count)

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
      utils.plot_loss_hist(content_loss_history, style_loss_history, total_loss_history)