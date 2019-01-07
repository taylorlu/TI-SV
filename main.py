import tensorflow as tf
import numpy as np
from model import ClassificationModel,AudioMeta
from tensorflow.python import pywrap_tensorflow
import sys
import os
import librosa

def train():

    lr = 0.002
    c = ClassificationModel()
    cost, normVector, alfas_mean = c.buildModel()

    gStep = tf.Variable(tf.constant(0))
    learning_rate = tf.train.exponential_decay(float(lr), gStep, 300, 0.9, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients, vars = zip(*optimizer.compute_gradients(cost))  #var_list=train_vars
    gradients, _ = tf.clip_by_global_norm(gradients, 100)
    train_op = optimizer.apply_gradients(zip(gradients, vars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver(var_list=restore_vars)    #var_list=restore_vars
        # saver.restore(sess, 'step/model.ckpt-'+ckpoint)
        saver = tf.train.Saver()

        for step in range(10000):
            mels, ids = c.getBatch_data_label(10)
            feed_dict = {c.melInputs:mels, c.labelInputs: ids, gStep: step}
            _, _lr, _loss, _alfas_mean = sess.run([train_op, learning_rate, cost, alfas_mean], feed_dict=feed_dict)
            print('{}: lr = {:.6f}, loss = {}, alfas_mean = {}'.format(step, _lr, _loss, _alfas_mean))

            if(step%300==0 and step!=0):
                saver.save(sess, 'step/model.ckpt', global_step=step)


def similar(matrix):
    ids = matrix.shape[0]
    for i in range(ids):
        for j in range(ids):
            # dist = matrix[i,:]*matrix[j,:]
            # dist = reduce(lambda x, y: x+y, dist)
            dist = np.linalg.norm(matrix[i,:] - matrix[j,:])
            print('%.2f  ' % dist, end='')
            if((j+1)%3==0 and j!=0):
                print("| ", end='')
        if((i+1)%3==0 and i!=0):
            print('\n')
            print("*"*80, end='')
        print("\n")


def test():
    c = ClassificationModel()
    _, normVector, _ = c.buildModel(is_training=False)

    restore_vars = []
    for var in tf.global_variables():
        if('temp' not in var.name):
            restore_vars.append(var)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=restore_vars)
        saver.restore(sess, 'step/model.ckpt-2700')

        speaker_path = '/home/logview/workspace/projects/TI-SV/samples'
        utterance_specs = []
        files = os.listdir(speaker_path)
        files.sort()
        for utter_name in files:

            utter_path = os.path.join(speaker_path, utter_name)         # path of each utterance
            utter, sr = librosa.core.load(utter_path, 16000)        # load utterance audio
            utter_trim, index = librosa.effects.trim(utter, top_db=20)  # voice activity detection, only trim


            S = librosa.feature.mfcc(y=utter_trim, sr=sr, n_mfcc=40)
            inputs = S.transpose((1,0))[:160]
            print(inputs.shape)
            utterance_specs.append(inputs)

        utterance_specs = np.array(utterance_specs)
        print(utterance_specs.shape)

        vectors = sess.run(normVector, feed_dict={c.melInputs:utterance_specs})
        similar(vectors)


if __name__ == '__main__':
    test()
