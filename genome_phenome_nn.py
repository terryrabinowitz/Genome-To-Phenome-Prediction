"""Builds the genotype2phenotype network.
"""

import tensorflow as tf
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
sys.path.append('/Users/terryrabinowitz/Desktop/cyclic-learning-rate-master/')
from clr import  cyclic_learning_rate
import 
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# -----Main Function-----
TRAIN = True  # train and validate model if True / test the model if False

# -----Training Parameters-----
BATCH_SIZE = 63  # 50
LEARN_RATE = 1.0e-2
L2_WEIGHT = 1.0e-2 # L2 regularization weight
EPOCH_CHECK = 1  # How often to save model and record loss / accuarcy values
EPOCH_LIMIT = 1000
CLIP = 0.0  # Clips gradiants to avoid overflow (not active if equal to 0)
NOISE = 0  # The standard deviation of gaussian noise that is added to the gradiants during back propogation (not active if equal to 0)
NUM_LAYERS_DENSE = 20
NUM_NODES = 10
DENSE_NUM_NODES = [NUM_NODES] * NUM_LAYERS_DENSE  # 800
DENSE_BATCH_NORM_FLAG = True
KEEP_PROB_DENSE = 1.0  # Dropout in the dense layers (1.0 = no dropout)
SCALE =1.0 #reduces the weight of the FIRST class label (used to reduce THC)
NUM_PCA = 70

# -----Data Parameters-----
GENERATE_DATA = False  # Random data used to test the models
DATA_SOURCE = "real"  # "random"
LOAD_TRAINING_MODEL = False  # Continue training from previously saved model
TRAIT_TARGET = "CANNABINOIDS"  # CANNABINOIDS, TERPENES, SPECIFIC, ALL
SPECIFIC_TRAIT_NUMBER = 0
# 0=THCA     1=d9THC    2=CBC 3=CBD 4=CBDA      5=CBDV      6=CBG 7=CBGA      8=CBN 9=d8THC     10=THCV
# 11=a-bisabool   12=a-humulene     13=a-pinene 14=a-terpinene    15=b-caryopholene 16=b-myrcene      17=b-pinene
# 18= camphene     19=caryopholene oxide   20=d3-carene      21=eucalyptol     22=g-terpinene    23=geraniol
# 24=isopulegol    25=limonene 26=nerolidol      27=ocimene  28=p-cymene 29=terpinolene
TEN_FOLD_VALIDATION = True  # ignore num samples for train, validate, test below if True
TEN_FOLD_VALIDATION_NUM = 7  # leaving 63 for training
TEN_FOLD_VALIDATION_COUNTER = 0

NUM_MARKERS = 1409  # 1000
NUM_STRANDS = 2
NUM_AMINO_ACIDS = 20

TAG = "str(TRAIT_TARGET)" + "_"
# -----Save Path-----
if TRAIT_TARGET == "CANNABINOIDS":
      NUM_TRAITS = 11  # 19  # Number of terpenes or cannabinoids (NOT INCLUDING TOTAL OR RELATIVE WEIGHTS)
      NUM_TRAITS_B = 10  # Number of terpenes or cannabinoids (NOT INCLUDING TOTAL OR RELATIVE WEIGHTS)
elif TRAIT_TARGET == "TERPENES":
      NUM_TRAITS = 19  # Number of terpenes or cannabinoids (NOT INCLUDING TOTAL OR RELATIVE WEIGHTS)
      NUM_TRAITS_B = 10  # Number of terpenes or cannabinoids (NOT INCLUDING TOTAL OR RELATIVE WEIGHTS)
elif TRAIT_TARGET == "SPECIFIC":
      TAG = TAG + str(SPECIFIC_TRAIT_NUMBER) + "_"
      NUM_TRAITS = 1  # Number of terpenes or cannabinoids (NOT INCLUDING TOTAL OR RELATIVE WEIGHTS)
      NUM_TRAITS_B = 1  # Number of terpenes or cannabinoids (NOT INCLUDING TOTAL OR RELATIVE WEIGHTS)
elif TRAIT_TARGET == "ALL":
      NUM_TRAITS = 30  # Number of terpenes or cannabinoids (NOT INCLUDING TOTAL OR RELATIVE WEIGHTS)
      NUM_TRAITS_B = 1  # Number of terpenes or cannabinoids (NOT INCLUDING TOTAL OR RELATIVE WEIGHTS)

# TAG = TAG + "attempt2_" + "_"
TAG = TAG + "num_layers_dense_" + str(NUM_LAYERS_DENSE) + "_"
TAG = TAG + "num_dense_nodes_" + str(DENSE_NUM_NODES[0]) + "_"
TAG = TAG + "dense_dropout_" + str(KEEP_PROB_DENSE) + "_"
TAG = TAG + "L2_" + str(L2_WEIGHT) + "_"
TAG = TAG + "batch_size" + str(BATCH_SIZE) + "_"
TAG = TAG + "lr_" + str(LEARN_RATE) + "_"
TAG = TAG + "val_set_" + str(TEN_FOLD_VALIDATION_COUNTER) + "/"

PATH_SAVE = "/Users/terryrabinowitz/PycharmProjects/cannabis/results/" + TAG

PATH_SAVE_TRAIN = PATH_SAVE + "train/"
if not os.path.exists(PATH_SAVE_TRAIN):
      os.makedirs(PATH_SAVE_TRAIN)
PATH_SAVE_VAL = PATH_SAVE + "val/"
if not os.path.exists(PATH_SAVE_VAL):
      os.makedirs(PATH_SAVE_VAL)
PATH_SAVE_FINAL = "/Users/terryrabinowitz/PycharmProjects/cannabis/results/final_results.txt"

# -----Data Path-----
PATH_DATA = "/Users/terryrabinowitz/PycharmProjects/cannabis/data/"


####################################################

def get_data():
      path_data = PATH_DATA
      if DATA_SOURCE == 'real':
            path_data += "real/"
            if TEN_FOLD_VALIDATION:
                  path_data_all_input = path_data + "ALL_INPUT.npy"
                  path_data_all_output = path_data + "ALL_OUTPUT.npy"
                  path_data_all_strains = path_data + "ALL_STRAINS.txt"
                  input_all = np.load(path_data_all_input)
                  input_all[input_all == 0] = -1 #change binary 0,1 to -1,1
                  print input_all.shape
                  input_all = input_all[:,0,:,:]
                  input_all = np.reshape(input_all,[70, 1409*20])
                  pca = PCA(n_components=NUM_PCA)
                  pca.fit(input_all)
                  print(pca.explained_variance_)
                  input_all = pca.transform(input_all)
                  print input_all.shape
                  # input_all_R = input_all[:, 0, :, :]
                  # input_all_R = np.reshape(input_all_R, [70, 1409 * 20])
                  # print input_all_R.shape
                  # path_save_all_input = path_data + "ALL_INPUT_R.npy"
                  # np.save(path_save_all_input, input_all_R)


                  if TRAIT_TARGET == "CANNABINOIDS":
                        output_all = np.load(path_data_all_output)[:, :NUM_TRAITS]
                  elif TRAIT_TARGET == "TERPENES":
                        output_all = np.load(path_data_all_output)[:, :NUM_TRAITS]
                  elif TRAIT_TARGET == "SPECIFIC":
                        output_all = np.expand_dims(np.load(path_data_all_output)[:, SPECIFIC_TRAIT_NUMBER], axis=1)
                  elif TRAIT_TARGET == "ALL":
                        output_all = np.load(path_data_all_output)
                  strains_all = []
                  f = open(path_data_all_strains, "r")
                  for i in f:
                        strains_all.append(i)

                  input_validate = input_all[
                                   TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM:TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM + TEN_FOLD_VALIDATION_NUM]
                  output_validate = output_all[
                                    TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM:TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM + TEN_FOLD_VALIDATION_NUM]
                  strain_validate = strains_all[
                                    TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM:TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM + TEN_FOLD_VALIDATION_NUM]

                  input_train = np.concatenate([input_all[:TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM],
                                                input_all[
                                                TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM + TEN_FOLD_VALIDATION_NUM:]])
                  output_train = np.concatenate([output_all[:TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM],
                                                 output_all[
                                                 TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM + TEN_FOLD_VALIDATION_NUM:]])
                  strain_train = np.concatenate([strains_all[:TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM],
                                                 strains_all[
                                                 TEN_FOLD_VALIDATION_COUNTER * TEN_FOLD_VALIDATION_NUM + TEN_FOLD_VALIDATION_NUM:]])
                  input_test = None
                  output_test = None
                  strain_test = None

            else:
                  path = path_data + "TRAIN_INPUT.npy"
                  input_train = np.load(path)
                  path = path_data + "TRAIN_OUTPUT.npy"
                  output_train = np.load(path)
                  path = path_data + "TRAIN_STRAINS.txt"
                  strain_train = []
                  f = open(path, "r")
                  for i in f:
                        strain_train.append(i)

                  path = path_data + "VALIDATE_INPUT.npy"
                  input_validate = np.load(path)
                  path = path_data + "VALIDATE_OUTPUT.npy"
                  output_validate = np.load(path)
                  path = path_data + "VALIDATE_STRAINS.txt"
                  strain_validate = []
                  f = open(path, "r")
                  for i in f:
                        strain_validate.append(i)

                  path = path_data + "TEST_INPUT.npy"
                  input_test = np.load(path)
                  path = path_data + "TEST_OUTPUT.npy"
                  output_test = np.load(path)
                  path = path_data + "TEST_STRAINS.txt"
                  strain_test = []
                  f = open(path, "r")
                  for i in f:
                        strain_test.append(i)
      return input_train, output_train, strain_train, input_validate, output_validate, strain_validate, input_test, output_test, strain_test


def shuffle_in_unison_2(a, b):
      rng_state = np.random.get_state()
      np.random.shuffle(a)
      np.random.set_state(rng_state)
      np.random.shuffle(b)


def shuffle_in_unison_3(a, b, c):
      rng_state = np.random.get_state()
      np.random.shuffle(a)
      np.random.set_state(rng_state)
      np.random.shuffle(b)
      np.random.set_state(rng_state)
      np.random.shuffle(c)


def shuffle_in_unison_5(a, b, c, d, e):
      rng_state = np.random.get_state()
      np.random.shuffle(a)
      np.random.set_state(rng_state)
      np.random.shuffle(b)
      np.random.set_state(rng_state)
      np.random.shuffle(c)
      np.random.set_state(rng_state)
      np.random.shuffle(d)
      np.random.set_state(rng_state)
      np.random.shuffle(e)


def cost_function_sse(pred, true):
      cost = tf.nn.l2_loss(pred - true)
      return cost


def cost_function_mse(pred, true, scale):
      cost = tf.losses.mean_squared_error(
            true,
            pred,
            weights=scale, #1
            scope=None,
            loss_collection=tf.GraphKeys.LOSSES,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
      )
      return cost


def layer (state, keep_prob_dense, layer_num):
      residual = state
      print state.get_shape()
      fan_in = int(state.get_shape()[1])        #70
      fan_out = DENSE_NUM_NODES[layer_num]  #600
      tag = "dense_weight_" + str(layer_num)
      weight = tf.Variable(tf.truncated_normal(shape=[fan_in, fan_out], stddev=0.1), name=tag)
      tag = "dense_bias_" + str(layer_num)
      bias = tf.Variable(tf.zeros([fan_out]), name=tag)
      state = tf.add(tf.matmul(state, weight), bias)
      if DENSE_BATCH_NORM_FLAG:
            batch_mean, batch_var = tf.nn.moments(state, [0])
            tag = "dense_norm_scale_" + str(layer_num)
            scale = tf.Variable(tf.ones([fan_out]), name=tag)
            tag = "dense_norm_beta_" + str(layer_num)
            beta = tf.Variable(tf.zeros([fan_out]), name=tag)
            state = tf.nn.batch_normalization(state, batch_mean, batch_var, beta, scale, variance_epsilon=1e-3)
      state = tf.nn.dropout(state, keep_prob=keep_prob_dense)
      state = tf.nn.relu(state)
      out = tf.concat([state, residual], axis=1)
      return out



def model_residual(state, keep_prob_dense):
      for layer_num in range(NUM_LAYERS_DENSE):
            state = layer(state, keep_prob_dense, layer_num)
      tag1 = "final_weight"
      tag2 = "final_bias"
      fan_in = int(state.get_shape()[1])
      fan_out = NUM_TRAITS
      weight = tf.Variable(
            tf.truncated_normal(shape=[fan_in, fan_out], stddev=0.1), name=tag1)
      bias = tf.Variable(tf.zeros([fan_out]), name=tag2)
      output = tf.add(tf.matmul(state, weight), bias)
      print output.get_shape()
      output = tf.nn.relu(output)
      return output



def model(state, keep_prob_dense):
      print state.get_shape()
      #state = tf.reshape(state, shape=[-1, NUM_STRANDS * NUM_AMINO_ACIDS * NUM_MARKERS])
      print state.get_shape()
      for layer_num in range(NUM_LAYERS_DENSE):
            if layer_num == 0:
                  fan_in = int(state.get_shape()[1])
            else:
                  fan_in = DENSE_NUM_NODES[layer_num - 1]
            fan_out = DENSE_NUM_NODES[layer_num]
            tag = "dense_weight_" + str(layer_num)
            weight = tf.Variable(tf.truncated_normal(shape=[fan_in, fan_out], stddev=0.1), name=tag)
            tag = "dense_bias_" + str(layer_num)
            bias = tf.Variable(tf.zeros([fan_out]), name=tag)
            state = tf.add(tf.matmul(state, weight), bias)
            if DENSE_BATCH_NORM_FLAG:
                  batch_mean, batch_var = tf.nn.moments(state, [0])
                  tag = "dense_norm_scale_" + str(layer_num)
                  scale = tf.Variable(tf.ones([fan_out]), name=tag)
                  tag = "dense_norm_beta_" + str(layer_num)
                  beta = tf.Variable(tf.zeros([fan_out]), name=tag)
                  state = tf.nn.batch_normalization(state, batch_mean, batch_var, beta, scale, variance_epsilon=1e-3)
            state = tf.nn.dropout(state, keep_prob=keep_prob_dense)
            state = tf.nn.relu(state)
            print state.get_shape()
      tag1 = "final_weight"
      tag2 = "final_bias"
      fan_in = int(state.get_shape()[1])
      fan_out = NUM_TRAITS  # the individual terpenes in absolute abundance
      weight = tf.Variable(
            tf.truncated_normal(shape=[fan_in, fan_out], stddev=0.1), name=tag1)
      bias = tf.Variable(tf.zeros([fan_out]), name=tag2)
      output = tf.add(tf.matmul(state, weight), bias)
      print output.get_shape()
      output = tf.nn.relu(output)
      return output





def train(input_train, output_train, strain_train, input_validate, output_validate, strain_validate):
      print
      if GENERATE_DATA:
            pass
            # print("Generating Input Data")
            # input_train = np.zeros((NUM_SAMPLES_TRAIN, NUM_STRANDS, NUM_MARKERS, NUM_AMINO_ACIDS))
            # for strain_counter in range(NUM_SAMPLES_TRAIN):
            #       for strand_counter in range(NUM_STRANDS):
            #             for snp_counter in range(NUM_MARKERS):
            #                   rand_aa = random.randint(0, 19)
            #                   input_train[strain_counter][strand_counter][snp_counter][rand_aa] = 1.0
            #
            # input_validate = np.zeros((NUM_SAMPLES_VALIDATE, NUM_STRANDS, NUM_MARKERS, NUM_AMINO_ACIDS))
            # for strain_counter in range(NUM_SAMPLES_VALIDATE):
            #       for strand_counter in range(NUM_STRANDS):
            #             for snp_counter in range(NUM_MARKERS):
            #                   rand_aa = random.randint(0, 19)
            #                   input_validate[strain_counter][strand_counter][snp_counter][rand_aa] = 1.0
            #
            # print("Generating Output Data")
            # output_train = np.random.rand(NUM_SAMPLES_TRAIN, NUM_TRAITS) * 5
            # output_validate = np.random.rand(NUM_SAMPLES_VALIDATE, NUM_TRAITS) * 5

      epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
      epoch_add_op = epoch.assign(epoch + 1)
      loaded_epoch = tf.placeholder(dtype=tf.int32)
      epoch_load_op = epoch.assign(loaded_epoch)


    #  learning_rate = tf.Variable(LEARN_RATE, trainable=False, dtype=tf.float32)
      learning_rate = cyclic_learning_rate(global_step=epoch, mode='triangular2')


      keep_prob_dense = tf.placeholder(dtype=tf.float32)
      L2_param = tf.constant(L2_WEIGHT, dtype=tf.float32)

      #input_variable = tf.placeholder(dtype=tf.float32, shape=[None, 2, NUM_MARKERS, 20])

      input_variable = tf.placeholder(dtype=tf.float32, shape=[None, NUM_PCA])
      output_variable = tf.placeholder(dtype=tf.float32, shape=[None, NUM_TRAITS])
      output_prediction_variable = model_residual(input_variable, keep_prob_dense)
     # cost = cost_function_sse(output_prediction_variable, output_variable)

      scaling = tf.placeholder(dtype=tf.float32, shape=[None, NUM_TRAITS])
      cost = cost_function_mse(output_prediction_variable, output_variable, scaling)
      total_parameters = 0
      L2_cost = 0
      counter = 0
      for variable in tf.trainable_variables():
            shape = variable.get_shape()
            L2_cost += tf.nn.l2_loss(variable)
            variable_parameters = 1
            for dim in shape:
                  variable_parameters *= dim.value
            total_parameters += variable_parameters
            counter += 1
      print "Num Variables = ", counter, "\tTotal parameters = ", total_parameters
      print

      L2_cost = tf.multiply(L2_param, L2_cost)
      cost = tf.add(cost, L2_cost)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1.0e-8)
      grads_and_vars = optimizer.compute_gradients(cost)
      if CLIP > 0:
            grads_and_vars = [(tf.clip_by_value(grad, -1 * CLIP, CLIP), var) for grad, var in grads_and_vars]
      if NOISE > 0:
            grads_and_vars = [(i[0] + tf.random_normal(shape=tf.shape(i[0]), mean=0, stddev=NOISE), i[1]) for i in
                              grads_and_vars]
      train_op = optimizer.apply_gradients(grads_and_vars)
      saver = tf.train.Saver()
      print("Running Session")
      with tf.Session() as sess:
            if LOAD_TRAINING_MODEL:
                  print("Reading model parameters from %s" % PATH_SAVE_TRAIN)
                  saver.restore(sess, tf.train.latest_checkpoint(PATH_SAVE_TRAIN))
                  tag = PATH_SAVE + "loss_summary.txt"
                  loss_file = open(tag)
                  previous_best = np.inf
                  counter = 1
                  for line in loss_file:
                        line = line.split()
                        lv = float(line[3])
                        if lv < previous_best:
                              previous_best = lv
                        counter += 1
                  loss_file.close()
                  sess.run(epoch_load_op, feed_dict={loaded_epoch: counter})
                  print "current epoch =", epoch.eval(session=sess)
                  print "best validation error =", previous_best
            else:
                  sess.run(tf.global_variables_initializer())
                  previous_best = np.inf
            best_accuracy_total = 0.0
            if TRAIT_TARGET == "CANNABINOIDS" or TRAIT_TARGET == "TERPENES" or TRAIT_TARGET == "ALL":
                  scale_train = np.array([[SCALE] + [1]*(NUM_TRAITS-1), ] * BATCH_SIZE)
                  scale_val = np.array([[SCALE] + [1]*(NUM_TRAITS-1), ] * len(input_validate))

            elif TRAIT_TARGET == "SPECIFIC":
                  scale_train = np.array([[SCALE] , ] * BATCH_SIZE)
                  scale_val = np.array([[SCALE] , ] * len(input_validate))

            while True:
                  sess.run(epoch_add_op)
                  loss_train = 0
                  loss_train_l2 = 0

                  shuffle_in_unison_3(input_train, output_train, strain_train)
                  for batch_num in xrange(0, len(input_train), BATCH_SIZE):
                        if batch_num + BATCH_SIZE > len(input_train):
                              break
                        input_train_batch = input_train[batch_num:batch_num + BATCH_SIZE]
                        output_train_batch = output_train[batch_num:batch_num + BATCH_SIZE]

                        _, loss, loss_l2, output_pred_train = sess.run(
                              [train_op, cost, L2_cost, output_prediction_variable],
                              feed_dict={output_variable: output_train_batch,
                                         input_variable: input_train_batch,
                                         scaling: scale_train,
                                         keep_prob_dense: KEEP_PROB_DENSE})
                        loss_train += loss
                        loss_train_l2 += loss_l2
                  loss_train -= loss_train_l2

                  loss_validate, loss_validate_l2, output_pred_validate = sess.run(
                        [cost, L2_cost, output_prediction_variable],
                        feed_dict={output_variable: output_validate,
                                   input_variable: input_validate,
                                   scaling: scale_val,
                                   keep_prob_dense: 1.0})
                  loss_validate -= loss_validate_l2
                  print epoch.eval(session=sess), "\t", loss_train, "\t", loss_validate
                  tag = PATH_SAVE_TRAIN + "best_train_loss_model"
                  saver.save(sess, tag)
                  if loss_validate < previous_best:
                        total_percent_real = np.sum(output_validate, axis=1)
                        for i in range(len(total_percent_real)):
                              if total_percent_real[i] == 0.0:
                                    total_percent_real[i] = 0.00001

                        total_percent_pred = np.sum(output_pred_validate, axis=1)
                        diff_total = abs(total_percent_real - total_percent_pred)
                        accuracy_total = (1.0 - diff_total / total_percent_real) * 100
                        tot = sum(accuracy_total)
                        accuracy_mean_total = tot / len(accuracy_total)
                        if accuracy_mean_total > best_accuracy_total:
                              best_accuracy_total = accuracy_mean_total
                        previous_best = loss_validate
                        tag = PATH_SAVE_VAL + "best_val_loss_model"
                        saver.save(sess, tag)
                  if epoch.eval(session=sess) % 25 == 0:
                        # diff = abs(output_validate - output_pred_validate)
                        # accuracy = (1.0 - diff / output_validate) * 100
                        # print accuracy
                        # exit()
                        total_percent_real = np.sum(output_validate, axis=1)
                        total_percent_pred = np.sum(output_pred_validate, axis=1)
                        diff_total = abs(total_percent_real - total_percent_pred)
                        accuracy_total = (1.0 - diff_total / total_percent_real) * 100
                        tot = sum(accuracy_total)
                        accuracy_mean_total = tot / len(accuracy_total)
                        if accuracy_mean_total > best_accuracy_total:
                              best_accuracy_total = accuracy_mean_total
                        print "\t\t", best_accuracy_total, "\t", accuracy_mean_total
                  if epoch.eval(session=sess) % EPOCH_CHECK == 0:
                        tag = PATH_SAVE + "loss_summary.txt"
                        string = str(loss_train) + " " + str(loss_validate) + "\n"
                        with open(tag, "a") as myfile:
                              myfile.write(string)
                  if epoch.eval(session=sess) == EPOCH_LIMIT:
                        break
            string = str(TRAIT_TARGET) + "\t" + str(BATCH_SIZE) + "\t" + str(NUM_LAYERS_DENSE) + "\t" + str(
                  NUM_NODES) + "\t" + str(LEARN_RATE) + "\t" + str(L2_WEIGHT) + "\t" + str(
                  epoch.eval(session=sess)) + "\t" + str(TEN_FOLD_VALIDATION_COUNTER) + "\t" + str(
                  best_accuracy) + "\t" + str(DENSE_BATCH_NORM_FLAG) + "\t" + str(KEEP_PROB_DENSE) + "\n"
            with open(PATH_SAVE_FINAL, "a") as myfile:
                  myfile.write(string)


def test():
      print
      if GENERATE_DATA:
            print("Generating Input Data")
            # input_test = np.zeros((NUM_SAMPLES_TEST, NUM_STRANDS, NUM_MARKERS, NUM_AMINO_ACIDS))
            # for strain_counter in range(NUM_SAMPLES_TEST):
            #       for strand_counter in range(NUM_STRANDS):
            #             for snp_counter in range(NUM_MARKERS):
            #                   rand_aa = random.randint(0, 19)
            #                   input_test[strain_counter][strand_counter][snp_counter][rand_aa] = 1.0

            input_test = np.zeros((NUM_SAMPLES_TEST, NUM_STRANDS, NUM_MARKERS, NUM_AMINO_ACIDS))
            # input_train[0] is all zero
            input_test[0][0][0][1] = 1.0
            input_test[0][1][0][1] = 1.0
            input_test[1][0][0][0] = 1.0
            input_test[1][1][0][0] = 0.0
      else:
            print("Loading Data")
            input_test = np.load(PATH_DATA_TEST_INPUT)
            output_test = np.load(PATH_DATA_TEST_OUTPUT)[:, :11]
            strain_test = []
            f = open(PATH_DATA_TEST_STRAIN, "r")
            for i in f:
                  strain_test.append(i)
      input_variable = tf.placeholder(dtype=tf.float32, shape=[NUM_SAMPLES_TEST, 2, NUM_MARKERS, 20])
      keep_prob_dense = tf.placeholder(dtype=tf.float32)
      output_prediction_variable = model_C(input_variable, keep_prob_dense)
      total_parameters = 0
      counter = 0
      print
      for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                  variable_parameters *= dim.value
            total_parameters += variable_parameters
            counter += 1
      print
      print "Num Variables = ", counter, "\tTotal parameters = ", total_parameters
      saver = tf.train.Saver()
      with tf.Session() as sess:
            print("Reading model parameters from %s" % PATH_SAVE_VAL)
            saver.restore(sess, tf.train.latest_checkpoint(PATH_SAVE_VAL))
            output_test_pred = sess.run(
                  [output_prediction_variable],
                  feed_dict={input_variable: input_test,
                             keep_prob_dense: 1.0})
            for i in range(len(output_test_pred[0])):
                  print strain_test[i], output_test[i], "\n", output_test_pred[0][i], "\n\n"


def main():
      train_input, train_output, train_strain, val_input, val_output, val_strain, test_input, test_output, test_strain = get_data()
      if TRAIN:
            train(train_input, train_output, train_strain, val_input, val_output, val_strain)
      else:
            test()


###########################

main()
