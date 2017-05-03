import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
import csv
import random
import sys

if (len(sys.argv)!=2):
  print("To train model please run \"python train_small_network.py [directory for training images]\"")
  exit()

TRAIN_DIR = sys.argv[1]

# used for scaling/normalization
IMAGE_SIZE = 224; 
CHANNELS = 3
pixel_depth = 255.0  

# For each LOADING BATCH
TRAINING_AND_VALIDATION_SIZE_DOGS = 625 
TRAINING_AND_VALIDATION_SIZE_CATS = 625 
TRAINING_AND_VALIDATION_SIZE_ALL  = 1250
TRAINING_SIZE = 1010 
VALID_SIZE = 240

#global list of files which are shuffled after every epoch
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

#Shuffle the global list of files after every epoch
def shuffle_data():
  random.shuffle(train_cats)
  random.shuffle(train_dogs)


###############FOR TENSORFLOW MODEL TRAINING#######################################################################
image_size = IMAGE_SIZE #redundant
num_labels = 2
batch_size = 16
num_channels = 3 # rgb
#For dropout
keep_prob_fc=0.5
keep_prob_cnn=0.9

#For model training on tensorflow. 
#Label format: [0.0 ,1.0] for dog and [1.0, 0.0] for cat
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  # set dogs to 1 and cats to 0
  labels = (labels=='dogs').astype(np.float32); 
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
############################################################################################################

# Reads th image and resizes it to IMAGE_SIZExIMAGE_SIZE while keeping aspect ratio by padding borders with black
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) 
    if (img.shape[0] >= img.shape[1]): # height is greater than width
       resizeto = (IMAGE_SIZE, int (round (IMAGE_SIZE * (float (img.shape[1])  / img.shape[0]))));
    else:
       resizeto = (int (round (IMAGE_SIZE * (float (img.shape[0])  / img.shape[1]))), IMAGE_SIZE);
    
    img2 = cv2.resize(img, (resizeto[1], resizeto[0]), interpolation=cv2.INTER_CUBIC)
    #for padding the border to keep the image a square
    img3 = cv2.copyMakeBorder(img2, 0, IMAGE_SIZE - img2.shape[0], 0, IMAGE_SIZE - img2.shape[1], cv2.BORDER_CONSTANT, 0)
        
    return img3[:,:,::-1]  # turn into rgb format

#Loads a set of images from disk to memory
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)

    for i, image_file in enumerate(images):
        image = read_image(image_file);
        image_data = np.array (image, dtype=np.float32);

        ##############IMAGE PREPROCESSING= mean subtraction and normalization####################
        image_data[:,:,0] -= np.mean(image_data[:,:,0])
        image_data[:,:,0] = (image_data[:,:,0].astype(float)) / pixel_depth
        image_data[:,:,1] -= np.mean(image_data[:,:,1])
        image_data[:,:,1] = (image_data[:,:,1].astype(float)) / pixel_depth
        image_data[:,:,2] -= np.mean(image_data[:,:,2])
        image_data[:,:,2] = (image_data[:,:,2].astype(float)) / pixel_depth
        #####################################################################################

        ##############PREPROCESSING= center each channel to 0 (-0.5,0.5)####################
        #image_data[:,:,0] = (image_data[:,:,0].astype(float) - pixel_depth / 2) / pixel_depth
        #image_data[:,:,1] = (image_data[:,:,1].astype(float) - pixel_depth / 2) / pixel_depth
        #image_data[:,:,2] = (image_data[:,:,2].astype(float) - pixel_depth / 2) / pixel_depth
        #####################################################################################
    	data[i] = image_data;
    	if i%250 == 0: print('Processed {} of {}'.format(i, count))    
    return data

#Randomizes the images in memory for splitting into training and validation sets
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

#For processing of each loading batch
def read_loading_batch(start_index):
	train_images = train_dogs[start_index:start_index+TRAINING_AND_VALIDATION_SIZE_DOGS] + train_cats[start_index:start_index+TRAINING_AND_VALIDATION_SIZE_CATS]
	train_labels = np.array ((['dogs'] * TRAINING_AND_VALIDATION_SIZE_DOGS) + (['cats'] * TRAINING_AND_VALIDATION_SIZE_CATS))
	train_normalized = prep_data(train_images)
	print("Train shape: {}".format(train_normalized.shape))
	train_dataset_rand, train_labels_rand = randomize(train_normalized, train_labels)

  #Splitting to validation and training set
	valid_dataset = train_dataset_rand[:VALID_SIZE,:,:,:]
	valid_labels =   train_labels_rand[:VALID_SIZE]
	train_dataset = train_dataset_rand[VALID_SIZE:VALID_SIZE+TRAINING_SIZE,:,:,:]
	train_labels  = train_labels_rand[VALID_SIZE:VALID_SIZE+TRAINING_SIZE]
	print ('Training', train_dataset.shape, train_labels.shape)

	#set up for tensorflow model training
	train_dataset, train_labels = reformat(train_dataset, train_labels)
	valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
	print ('Training set for tensorflow', train_dataset.shape, train_labels.shape)
	print ('Validation set for tensorflow', valid_dataset.shape, valid_labels.shape)
	return train_dataset, train_labels, valid_dataset, valid_labels


graph = tf.Graph()

#Model set up
#Small network
#INPUT->[CONV->RELU->POOL]*3->FC->RELU->FC
with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.placeholder(
    tf.float32, shape=(VALID_SIZE, image_size, image_size, num_channels))
  tf_valid_labels = tf.placeholder(tf.float32, shape=(VALID_SIZE, num_labels))

  ##### variables 
  #Set 1 of [CONV->RELU->POOL]
  kernel_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32,
                                            stddev=1e-1), name='weights_conv1')
  biases_conv1 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                        trainable=True, name='biases_conv1')
  #Set 2 of [CONV->RELU->POOL]
  kernel_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], dtype=tf.float32,
                                            stddev=1e-1), name='weights_conv2')
  biases_conv2 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                        trainable=True, name='biases_conv2')
  #Set 3 of [CONV->RELU->POOL]
  kernel_conv3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32,
                                            stddev=1e-1), name='weights_conv3')
  biases_conv3 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                        trainable=True, name='biases_conv3')
  #For FC layer
  fc1w = tf.Variable(tf.truncated_normal([50176, 64], 
                                                dtype=tf.float32,
                                                stddev=1e-1), name='weights')
  fc1b = tf.Variable(tf.constant(1.0, shape=[64], dtype=tf.float32),
                        trainable=True, name='biases')
  #For final output
  fc2w = tf.Variable(tf.truncated_normal([64, 2],
                                                dtype=tf.float32,
                                                stddev=1e-1), name='weights')
  fc2b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),
                        trainable=True, name='biases')
 
  
  def model(data, training):
    #For conv set 1
     with tf.name_scope('conv1_1') as scope:
         conv = tf.nn.conv2d(data, kernel_conv1, [1, 1, 1, 1], padding='SAME')
         out = tf.nn.bias_add(conv, biases_conv1)
         conv1_1 = tf.nn.relu(out, name=scope)
         
     # pool1
     pool1 = tf.nn.max_pool(conv1_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool1')
     #Dropout
     if (training):
     	pool1 = tf.nn.dropout(pool1,keep_prob_cnn)
     
    #For conv set 2
     with tf.name_scope('conv2_1') as scope:
         conv = tf.nn.conv2d(pool1, kernel_conv2, [1, 1, 1, 1], padding='SAME')
         out = tf.nn.bias_add(conv, biases_conv2)
         conv2_1 = tf.nn.relu(out, name=scope)
         
     # pool2
     pool2 = tf.nn.max_pool(conv2_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool2')
     #Dropout
     if (training):
     	pool2 = tf.nn.dropout(pool2, keep_prob_cnn)

    #For conv set 3
     with tf.name_scope('conv3_1') as scope:
         conv = tf.nn.conv2d(pool2, kernel_conv3, [1, 1, 1, 1], padding='SAME')
         out = tf.nn.bias_add(conv, biases_conv3)
         conv3_1 = tf.nn.relu(out, name=scope)
         
     # pool3
     pool3 = tf.nn.max_pool(conv3_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool3')
     #Dropout
     if (training):
     	pool3 = tf.nn.dropout(pool3, keep_prob_cnn)
         
     # fc1
     with tf.name_scope('fc1') as scope:
         shape = int(np.prod(pool3.get_shape()[1:])) 
         pool3_flat = tf.reshape(pool3, [-1, shape])
         fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
         fc1 = tf.nn.relu(fc1l)

     if (training):    
     	fc1 = tf.nn.dropout(fc1, keep_prob_fc)

     #  output
     with tf.name_scope('fc2') as scope:
         fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
     return fc2l;
  
  # Training computation.
  logits = model(tf_train_dataset, True)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset, False))
  valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=valid_prediction, labels=tf_valid_labels))

def accuracy(predictions, labels):
   return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

num_steps = int(TRAINING_SIZE/batch_size) #number of iterations for each LOADING BATCH
num_epochs = 15
batch_sets = int(12500/TRAINING_AND_VALIDATION_SIZE_CATS) #number of loading batches


with tf.Session(graph=graph) as session:
  #writer= tf.summary.FileWriter('./graphs',session.graph)
  saver =  tf.train.Saver()
  tf.initialize_all_variables().run()
  print ("Initialized")

  #####For training continuation of saved model
  #saver.restore(session, "./model_rm.ckpt")
  #print ("Model restored!")

  #For graphing of results
  y_valid_accuracy=[]
  y_valid_loss=[]
  y_training_loss=[]
  x=[]


  for epoch in range(num_epochs):
  	for loading_batch in range(batch_sets):
  		train_dataset, train_labels, valid_dataset, valid_labels = read_loading_batch(loading_batch*TRAINING_AND_VALIDATION_SIZE_CATS)
  		print("Data Loaded \n\n")

  		for i in range(1):
			for step in range(num_steps):
				offset = (step * batch_size)
				batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
				batch_labels = train_labels[offset:(offset + batch_size), :]
				feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_valid_dataset : valid_dataset}
				_, l, predictions = session.run(
				[optimizer, loss, train_prediction], feed_dict=feed_dict)

        #print and graph current status every 21st iteration
				if ((step+1) % 21 == 0):
					feed_dict_test={tf_valid_dataset : valid_dataset, tf_valid_labels: valid_labels}
					print ("Minibatch loss at batch:\t", loading_batch, "step:\t",step, "loss:\t", l)
					print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
					v_prediction, v_loss = session.run([valid_prediction, valid_loss], feed_dict=feed_dict_test)
					print ("Validation accuracy: %.1f%%" % accuracy(v_prediction, valid_labels))
					print ("Validation loss: ", v_loss)
					x.append(len(x)*21)
					y_valid_accuracy.append(accuracy(v_prediction, valid_labels))
					y_valid_loss.append(v_loss)
					y_training_loss.append(l)

  #Save model after every epoch
	save_path = saver.save(session, "./models/trial.ckpt")
	print("Model saved in file: %s\n" % save_path)

  #Output graphs
	plt.clf()
	plt.plot(x,y_valid_accuracy)
	plt.xlabel('Iterations')
	plt.ylabel('Accuracy')
	plt.savefig('validation_accuracy_small_network.png')

	plt.clf()
	plt.plot(x,y_valid_loss)
	plt.xlabel('Iterations')
	plt.ylabel("Validation cross entropy loss")
	plt.savefig('validation_loss_small_network.png')

	plt.clf()
	plt.plot(x,y_training_loss)
	plt.xlabel('Iterations')
	plt.ylabel('Training minibatch loss')
	plt.savefig('training_miniloss_small_network.png')
