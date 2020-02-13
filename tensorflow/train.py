%tensorflow_version 2.x
import tensorflow as tf

import random
import numpy as np
import time

import vgg
import transformer
import utils

!rm -rf tf_utils
!git clone https://github.com/mixuala/tf_utils.git
from tf_utils import helpers
from tf_utils.io import load, show, save

# GLOBAL SETTINGS
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "/content/train"
NUM_EPOCHS = 1
STYLE_IMAGE_PATH = "images/mosaic.jpg"
BATCH_SIZE = 4 
CONTENT_WEIGHT = 17 # 17
STYLE_WEIGHT = 50 # 25
ADAM_LR = 0.001
SAVE_MODEL_PATH = "models/"
SAVE_IMAGE_PATH = "images/out/"
SAVE_MODEL_EVERY = 500 # 2,000 Images with batch size 4
SEED = 35
PLOT_LOSS = 1

def train():
  # # Seeds
  # torch.manual_seed(SEED)
  # torch.cuda.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  # # Device
  # device = ("cuda" if torch.cuda.is_available() else "cpu")

  # # Dataset and Dataloader
  def transforms(filename):
    parts = tf.strings.split(filename, '/')
    label = parts[-2]
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = random_sq_crop(image, size=256, margin_pct=5)
    image = tf.image.resize( image, (TRAIN_IMAGE_SIZE,TRAIN_IMAGE_SIZE))
    return image, label
  # transform = transforms.Compose([
  # transforms.Resize(TRAIN_IMAGE_SIZE),
  # transforms.CenterCrop(TRAIN_IMAGE_SIZE),
  # transforms.ToTensor(),
  # transforms.Lambda(lambda x: x.mul(255))
  # ])

  # train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
  # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  list_ds = tf.data.Dataset.list_files('{}/*/*'.format(DATASET_PATH))
  train_dataset = list_ds.map(transforms).shuffle(buffer_size=100).batch(BATCH_SIZE)

  # # Load networks
  TransformerNetwork = transformer.TransformerNetwork()
  style_model = vgg.layers["vgg19_torch"]
  VGG = vgg.vgg_layers0( style_model['content_layers'], style_model['style_layers'] )

  # # Get Style Features, BGR ordering
  rgb_mean = tf.reshape(tf.constant( [123.68, 116.779, 103.939], dtype=tf.float32), (1,1,1,3))
  rgb_mean_normalized = tf.reshape(tf.constant( [0.48501961, 0.45795686, 0.40760392], dtype=tf.float32), (1,1,1,3))

  # style_image is static, cache features
  style_image = load(STYLE_IMAGE_PATH)[...,:3] # tf.float32
  style_tensor = tf.subtract( style_image[tf.newaxis, ...], rgb_mean_normalized )
  # B, H, W, C = style_tensor.shape
  (_, style_features) = VGG( tf.repeat( style_tensor, repeats=BATCH_SIZE, axis=0), preprocess=True )
  target_style_gram = [ utils.gram(value)  for value in style_features ]  # list

  # # Optimizer settings
  optimizer = tf.optimizers.Adam(learning_rate=ADAM_LR, beta_1=0.99, epsilon=1e-1)
  # optimizer = optim.Adam(TransformerNetwork.parameters(), lr=ADAM_LR)

  # # Loss trackers
  content_loss_history = []
  style_loss_history = []
  total_loss_history = []
  batch_content_loss_sum = 0
  batch_style_loss_sum = 0
  batch_total_loss_sum = 0

  # # Optimization/Training Loop
  @tf.function()
  def train_step(x_train, y_train=None, loss_weights=None, log_freq=10):
    with tf.GradientTape() as tape:
      # Generate images and get features
      # content_batch = content_batch[:,[2,1,0]].to(device) # 
      generated_batch = TransformerNetwork(x_train)
      # TODO: clip and/or tanh before VGG()??, what is the domain of generated_batch?
      (generated_content_features, generated_style_features) = VGG( generated_batch, preprocess=True )
      generated_style_gram = [ utils.gram(value)  for value in generated_style_features ]  # list
      (content_features, _) = VGG(x_train, preprocess=True )

      # Content Loss
      MSELoss = tf.keras.losses.MSE
      for (y_true, y_pred) in zip(generated_content_features, content_features):
        content_loss = CONTENT_WEIGHT * MSELoss(y_true, y_pred)            
        batch_content_loss_sum += content_loss

      # Style Loss
      style_loss = 0
      for (y_true, y_pred)in zip(target_style_gram, generated_style_gram):
        s_loss = MSELoss(y_true, y_pred)
        style_loss += s_loss
      style_loss *= STYLE_WEIGHT
      batch_style_loss_sum += style_loss

      # Total Loss
      total_loss = content_loss + style_loss
      batch_total_loss_sum += total_loss  # ???: how do I use this with GradientTape?
      # end: with tf.GradientTape()

    # apply batch_losses to grads
    grads = tape.gradient(total_loss, TransformerNetwork.trainable_weights)
    optimizer.apply_gradients(zip(grads, TransformerNetwork.trainable_weights))
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
      # Get current batch size in case of odd batch sizes
      curr_batch_size = content_batch.shape[0]

      #   # Backprop and Weight Update
      train_step(content_batch)

    # Save Model and Print Losses
    if (((batch_count-1)%SAVE_MODEL_EVERY == 0) or (batch_count==NUM_EPOCHS*len(train_loader))):
      # Print Losses
      print("========Iteration {}/{}========".format(batch_count, NUM_EPOCHS*len(train_loader)))
      print("\tContent Loss:\t{:.2f}".format(batch_content_loss_sum/batch_count))
      print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum/batch_count))
      print("\tTotal Loss:\t{:.2f}".format(batch_total_loss_sum/batch_count))
      print("Time elapsed:\t{} seconds".format(time.time()-start_time))

      # # Save Model
      # checkpoint_path = SAVE_MODEL_PATH + "checkpoint_" + str(batch_count-1) + ".pth"
      # torch.save(TransformerNetwork.state_dict(), checkpoint_path)
      # print("Saved TransformerNetwork checkpoint file at {}".format(checkpoint_path))

      # Save sample generated image
      sample_tensor = generated_batch[0].clone().detach().unsqueeze(dim=0)
      # ???: domain=(-1.4763, 508.0564), img is clipped to (0,255) in utils.saveimg
      print( "\tImage, domain:\t({:.1f},{:.1f})".format( torch.min(sample_tensor).numpy(), torch.max(sample_tensor).numpy() ) )
      sample_image = utils.ttoi(sample_tensor.clone().detach())
      sample_image_path = SAVE_IMAGE_PATH + "sample0_" + str(batch_count-1) + ".png"
      utils.saveimg(sample_image, sample_image_path)
      print("Saved sample tranformed image at {}".format(sample_image_path))

      # Save loss histories
      content_loss_history.append(batch_total_loss_sum/batch_count)
      style_loss_history.append(batch_style_loss_sum/batch_count)
      total_loss_history.append(batch_total_loss_sum/batch_count)

    # Iterate Batch Counter
    batch_count+=1