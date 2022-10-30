# -*- coding: utf-8 -*-
'''
2022-05-01 Weizheng Yan conanywz@gmail.com
ersion: v0.1
'''

from yan_utilities import *
import scipy.io as scio
import numpy as np
import pandas as pd
import os
import time
import datetime
import sys
import mat73
from os.path import abspath, join, dirname, exists
try:
    AUTOTUNE = tf.data.AUTOTUNE
except:
    AUTOTUNE = tf.data.experimental.AUTOTUNE

from tensorflow.python.keras.backend import set_session
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))
print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

sys.path.insert(0, join(abspath(dirname(__file__)), './CombatHarmonization'))
from ComBatHarmonization.Python.neuroCombat import neuroCombat

class ComBat(object):

    def __init__(self, list):
        self.feature_name = list  # 'Demo' 'MRI3D'  'Cortical_thickness'
        self.harmonization_mode = 'ComBat'
        self.data_name = 'result/' + 'Origin' + '_' + self.feature_name + '_data_category_domain.mat'

        if self.feature_name not in ['MRI3D']:
            self.data_name_to_save = 'result/' + self.harmonization_mode + '_' + self.feature_name + '_data_category_domain.mat'
            try:
                dataset = mat73.loadmat(self.data_name)
            except:
                dataset = scio.loadmat(self.data_name)

            self.data = dataset['data']
            self.domain_label = np.squeeze(dataset['domain_label'])
            self.category_label = np.squeeze(dataset['category_label'])

        if self.feature_name in ['MRI3D']:
            self.data_name = 'result/' + 'Origin_' + self.feature_name + '_data_category_domain' + '.mat'
            self.domain_selected = [2, 4]
            data_domain1, data_domain2, category_label_domain1_one_hot, category_label_domain2_one_hot = data_preprocessing_and_domain_selection(
                self.data_name, self.domain_selected[0], self.domain_selected[1], output_label_one_hot=True, feature_type='MRI3D')

            self.data_domain1_Origin = 'data/T1_data/' + pd.DataFrame(data_domain1)[0].to_numpy() + '.npy'
            self.data_domain2_Origin = 'data/T1_data/' + pd.DataFrame(data_domain2)[0].to_numpy() + '.npy'
            self.data_domain1_ComBat = 'data/T1_data/' + pd.DataFrame(data_domain1)[0].to_numpy() + '_ComBat.npy'
            self.data_domain2_ComBat = 'data/T1_data/' + pd.DataFrame(data_domain2)[0].to_numpy() + '_ComBat.npy'

    def get_data_from_each_domain(self):

        with h5py.File('result/ComBat_MRI3D_before.h5','w') as f:

            data_domain1 = []
            data_domain2 = []

            for index in range(len(self.data_domain1_Origin)):
                temp = np.load(self.data_domain1_Origin[index])[12: -13, 8: -9, 12: -13]
                print('Index in domain 1 is: ',index)
                temp = temp.flatten()
                data_domain1.append(temp)

            for index in range(len(self.data_domain2_Origin)):
                temp = np.load(self.data_domain2_Origin[index])[12: -13, 8: -9, 12: -13]
                print('Index in domain 2 is: ',index)
                temp = temp.flatten()
                data_domain2.append(temp)

            f.create_dataset('data_domain1',data=np.array(data_domain1))
            f.create_dataset('data_domain2',data=np.array(data_domain2))

    def apply_combat(self):

        if self.feature_name not in ['MRI3D']:
            # Now, use the ComBat for harmonization.
            covars = {'batch': self.domain_label}
            covars = pd.DataFrame(covars)
            batch_col = 'batch'
            data_combat = neuroCombat(dat=np.transpose(self.data),
                                      covars=covars,
                                      batch_col=batch_col,
                                      categorical_cols=None)["data"]

            domain_label = np.expand_dims(self.domain_label, axis=1)
            category_label = np.expand_dims(self.category_label, axis=1)
            scio.savemat(self.data_name_to_save, {'data': np.transpose(data_combat), 'domain_label': domain_label,
                                             'category_label': category_label})
            print('ok')

        if self.feature_name in ['MRI3D']:

            if not exists('result/ComBat_MRI3D_before.h5'):
                self.get_data_from_each_domain()

            with h5py.File('result/ComBat_MRI3D_before.h5','r') as f:
                data_domain1 = f['data_domain1'][:]
                data_domain2 = f['data_domain2'][:]

            data_domain1 = np.array(data_domain1)
            data_domain2 = np.array(data_domain2)
            batch_list = [1]*len(data_domain1) + [2]*len(data_domain2)
            data_domains = np.concatenate([data_domain1, data_domain2], axis=0)
            del data_domain1
            del data_domain2

            data_domains = np.transpose(data_domains)

            with h5py.File('result/ComBat_MRI3D_after.h5','w') as f:

                covars = {'batch': batch_list}
                covars = pd.DataFrame(covars)
                batch_col = 'batch'

                step_range = 20000
                data_combat = []
                for i in range(0,96*128*96,step_range):
                    if i+step_range < 96*128*96:
                        data_domains_part = data_domains[i:i+step_range,:]
                    else:
                        data_domains_part = data_domains[i:, :]

                    data_combat_part = neuroCombat(dat=data_domains_part,
                                              covars=covars,
                                              batch_col=batch_col,
                                              categorical_cols=None)["data"]
                    data_combat.append(data_combat_part)

                del data_domains
                data_combat = np.concatenate(data_combat, axis=0)
                data_combat = np.transpose(data_combat)
                f.create_dataset('data_combat', data=data_combat)

    def save_combat_to_each_subject(self):

        if self.feature_name in ['MRI3D']:

            with h5py.File('result/ComBat_MRI3D_after.h5','r') as f:
                data_combat = f['data_combat'][:]

            temp_mask = np.zeros([121, 145, 121])

            for index in range(len(data_combat)):
                temp = data_combat[index]
                temp = np.reshape(temp, [96, 128, 96])

                temp_mask[12: -13, 8: -9, 12: -13] = temp

                if index in range(len(self.data_domain1_Origin)):
                    save_name = self.data_domain1_Origin[index]
                    np.save(save_name[:-4] + '_ComBat' + '.npy', temp_mask)
                if index in range(len(self.data_domain1_Origin),len(self.data_domain1_Origin) + len(self.data_domain2_Origin)):
                    save_name = self.data_domain2_Origin[index-len(self.data_domain1_Origin)]
                    np.save(save_name[:-4] + '_ComBat' + '.npy', temp_mask)

class CycleGAN(object):
    def __init__(self, list):
        self.feature_name = list  # 'Demo'    #'MRI3D'   #'Cortical_thickness'
        self.harmonization_mode = 'CycleGAN'

        if self.feature_name in ['Demo', 'DemoCircle']:
            self.feature_type = 'vector'
            self.DL_model_name = 'DemoDNN'
            self.BUFFER_SIZE = 1200
            self.BATCH_SIZE = 100
            self.input_vector_size = 2
            self.EPOCHS = 400
            self.domain_selected = [1, 2]
            self.lambda_cycle_loss_control = 2

        if self.feature_name in ['Cortical_thickness']:
            self.feature_type = 'vector'
            self.DL_model_name = 'DNN'
            self.BUFFER_SIZE = 10000  # must over the sample amounts.
            self.BATCH_SIZE = 100
            self.input_vector_size = 68
            self.EPOCHS = 300
            self.domain_selected = [2, 4] # select two domains
            self.lambda_cycle_loss_control = 5

        if self.feature_name in ['MRI3D']:
            self.feature_type = 'MRI3D'
            self.DL_model_name = '3DCNN'
            self.BUFFER_SIZE = 1100
            self.BATCH_SIZE = 16
            self.input_vector_size = [96,128,96,1]
            self.EPOCHS = 100
            self.domain_selected = [2, 4] #
            self.lambda_cycle_loss_control = 5

        self.data_name = 'result/' + 'Origin_' + self.feature_name + '_data_category_domain'+ '.mat'

        self.use_checkpoints = True

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'result/harmonization/logs_'+ self.feature_name +'/' + self.harmonization_mode + '_domain_selected_' + str(self.domain_selected[0])+str(self.domain_selected[1])+ '_' + self.current_time
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.checkpoint_path = "result/harmonization/checkpoints_" + self.feature_name +'/' + self.harmonization_mode + '_domain_selected_' + str(self.domain_selected[0])+str(self.domain_selected[1])

        data_domain1, data_domain2, category_label_domain1_one_hot, category_label_domain2_one_hot = data_preprocessing_and_domain_selection(
            self.data_name, self.domain_selected[0], self.domain_selected[1], output_label_one_hot=True,
            feature_type=self.feature_type)

        if self.feature_name not in ['MRI3D']:
            self.domain1 = tf.data.Dataset.from_tensor_slices((np.float32(data_domain1), category_label_domain1_one_hot))
            self.domain1 = self.domain1.cache().shuffle(self.BUFFER_SIZE, seed=10, reshuffle_each_iteration=True).batch(batch_size=self.BATCH_SIZE)
            self.domain2 = tf.data.Dataset.from_tensor_slices((np.float32(data_domain2), category_label_domain2_one_hot))
            self.domain2 = self.domain2.cache().shuffle(self.BUFFER_SIZE, seed=10, reshuffle_each_iteration=True).batch(batch_size=self.BATCH_SIZE)

        if self.feature_name in ['MRI3D']:
            data_domain1 = 'data/T1_data/' + pd.DataFrame(data_domain1)[0].to_numpy() + '.npy'
            data_domain2 = 'data/T1_data/' + pd.DataFrame(data_domain2)[0].to_numpy() + '.npy'
            self.domain1 = tf.data.Dataset.from_tensor_slices((data_domain1, np.float32(category_label_domain1_one_hot)))
            self.domain1 = self.domain1.map(map_func=read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(self.BUFFER_SIZE, seed=10, reshuffle_each_iteration=True).batch(batch_size=self.BATCH_SIZE)
            self.domain2 = tf.data.Dataset.from_tensor_slices((data_domain2, np.float32(category_label_domain2_one_hot)))
            self.domain2 = self.domain2.map(map_func=read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(self.BUFFER_SIZE, seed=10, reshuffle_each_iteration=True).batch(batch_size=self.BATCH_SIZE)

        self.n = 0

        print('## feature name: ', self.feature_name)
        print('Current time is: ', self.current_time)

        self.generator_g_scan = get_generator(generator_name=self.DL_model_name, input_vector_size=self.input_vector_size)
        self.generator_f_scan = get_generator(generator_name=self.DL_model_name, input_vector_size=self.input_vector_size)
        self.discriminator_x_scan = get_discriminator(discriminator_name=self.DL_model_name, input_vector_size=self.input_vector_size)
        self.discriminator_y_scan = get_discriminator(discriminator_name=self.DL_model_name, input_vector_size=self.input_vector_size)
        self.classifier_scan_F1 = get_classifier(classifier_name=self.DL_model_name, input_vector_size=self.input_vector_size)

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.classifier_optimizer_F1 = tf.keras.optimizers.Adam()

        self.ckpt = tf.train.Checkpoint(generator_g=self.generator_g_scan,
                                   generator_f=self.generator_f_scan,
                                   discriminator_x=self.discriminator_x_scan,
                                   discriminator_y=self.discriminator_y_scan,
                                   classifier_scan_F1=self.classifier_scan_F1,
                                   generator_g_optimizer=self.generator_g_optimizer,
                                   generator_f_optimizer=self.generator_f_optimizer,
                                   discriminator_x_optimizer=self.discriminator_x_optimizer,
                                   discriminator_y_optimizer=self.discriminator_y_optimizer,
                                   classifier_optimizer_F1=self.classifier_optimizer_F1,
                                   )
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=1000)
        if self.use_checkpoints and self.ckpt_manager.latest_checkpoint:  # if a checkpoint exists and we want to use it, restore the latest checkpoint.
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest harmonization checkpoint restored==========!!')

        self.tb_gen_g_loss = tf.keras.metrics.Mean('gen_g_loss', dtype=tf.float32)
        self.tb_gen_f_loss = tf.keras.metrics.Mean('gen_f_loss', dtype=tf.float32)
        self.tb_disc_x_loss = tf.keras.metrics.Mean('disc_x_loss', dtype=tf.float32)
        self.tb_disc_y_loss = tf.keras.metrics.Mean('disc_y_loss', dtype=tf.float32)
        self.tb_class_domain1_batch_data_F1_loss = tf.keras.metrics.Mean('class_domain1_batch_data_F1_loss', dtype=tf.float32)

        self.tb_total_cycle_loss = tf.keras.metrics.Mean('total_cycle_loss', dtype=tf.float32)
        self.tb_total_gen_g_loss = tf.keras.metrics.Mean('total_gen_g_loss', dtype=tf.float32)
        self.tb_total_gen_f_loss = tf.keras.metrics.Mean('total_gen_f_loss', dtype=tf.float32)

        self.tb_ACC_metrics_domain1_batch_data_F1 = tf.keras.metrics.Mean('ACC_domain1_batch_data_F1',dtype=tf.float32)
        self.tb_ACC_metrics_domain2_batch_data_F1 = tf.keras.metrics.Mean('ACC_test', dtype=tf.float32)

        self.ACC_metrics_domain1_batch_data_F1 = tf.keras.metrics.Accuracy()
        self.ACC_metrics_domain2_batch_data_F1 = tf.keras.metrics.Accuracy()

    @tf.function
    def train_step(self, domain1_batch_data, domain2_batch_data,domain1_batch_label, domain2_batch_label):
        with tf.GradientTape(persistent=True) as tape:  # persistent is set to True because the tape is used more than once to calculate the gradients.
            fake_domain2_batch_data = self.generator_g_scan(domain1_batch_data, training=True)  # Generator G translates X -> Y
            cycled_domain1_batch_data = self.generator_f_scan(fake_domain2_batch_data, training=True)  # Generator F translates Y -> X
            fake_domain1_batch_data = self.generator_f_scan(domain2_batch_data, training=True)
            cycled_domain2_batch_data = self.generator_g_scan(fake_domain1_batch_data, training=True)

            same_domain1_batch_data = self.generator_f_scan(domain1_batch_data, training=True)  # same_domain1_batch_data and same_domain2_batch_data are used for identity loss.
            same_domain2_batch_data = self.generator_g_scan(domain2_batch_data, training=True)

            disc_domain1_batch_data = self.discriminator_x_scan(domain1_batch_data, training=True)
            disc_domain2_batch_data = self.discriminator_y_scan(domain2_batch_data, training=True)

            disc_fake_domain1_batch_data = self.discriminator_x_scan(fake_domain1_batch_data, training=True)
            disc_fake_domain2_batch_data = self.discriminator_y_scan(fake_domain2_batch_data, training=True)

            classified_domain1_batch_data_F1 = self.classifier_scan_F1(cycled_domain1_batch_data, training=True)
            classified_fake_domain1_batch_data_F1 = self.classifier_scan_F1(fake_domain1_batch_data, training=True)

            gen_g_loss = generator_loss(disc_fake_domain2_batch_data)
            gen_f_loss = generator_loss(disc_fake_domain1_batch_data)
            total_cycle_loss = calc_cycle_loss(domain1_batch_data, cycled_domain1_batch_data, self.lambda_cycle_loss_control) + calc_cycle_loss(domain2_batch_data,cycled_domain2_batch_data,self.lambda_cycle_loss_control)

            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(domain2_batch_data, same_domain2_batch_data,self.lambda_cycle_loss_control)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(domain1_batch_data, same_domain1_batch_data,self.lambda_cycle_loss_control)

            disc_x_loss = discriminator_loss(disc_domain1_batch_data, disc_fake_domain1_batch_data)
            disc_y_loss = discriminator_loss(disc_domain2_batch_data, disc_fake_domain2_batch_data)

            class_domain1_batch_data_F1_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(domain1_batch_label, classified_domain1_batch_data_F1))
            total_class_loss = class_domain1_batch_data_F1_loss

            self.ACC_metrics_domain1_batch_data_F1.update_state(tf.math.argmax(domain1_batch_label, 1),tf.math.argmax(classified_domain1_batch_data_F1, 1))
            self.ACC_metrics_domain2_batch_data_F1.update_state(tf.math.argmax(domain2_batch_label, 1),tf.math.argmax(classified_fake_domain1_batch_data_F1, 1))

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g_scan.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f_scan.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, self.generator_g_scan.trainable_variables))
        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, self.generator_f_scan.trainable_variables))

        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x_scan.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y_scan.trainable_variables)
        classifier_gradients_F1 = tape.gradient(total_class_loss, self.classifier_scan_F1.trainable_variables)

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, self.discriminator_x_scan.trainable_variables))
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, self.discriminator_y_scan.trainable_variables))

        self.classifier_optimizer_F1.apply_gradients(zip(classifier_gradients_F1, self.classifier_scan_F1.trainable_variables))

        self.tb_gen_g_loss(gen_g_loss)
        self.tb_gen_f_loss(gen_f_loss)
        self.tb_total_cycle_loss(total_cycle_loss)
        self.tb_total_gen_g_loss(total_gen_g_loss)
        self.tb_total_gen_f_loss(total_gen_f_loss)
        self.tb_disc_x_loss(disc_x_loss)
        self.tb_disc_y_loss(disc_y_loss)
        self.tb_class_domain1_batch_data_F1_loss(class_domain1_batch_data_F1_loss)
        self.tb_ACC_metrics_domain1_batch_data_F1(self.ACC_metrics_domain1_batch_data_F1.result())
        self.tb_ACC_metrics_domain2_batch_data_F1(self.ACC_metrics_domain2_batch_data_F1.result())

    def train(self):
        for epoch in range(self.EPOCHS):
            start = time.time()
            print('Epoch: ',epoch)

            for domain1_batch, domain2_batch in tf.data.Dataset.zip((self.domain1, self.domain2)):
                self.train_step(domain1_batch[0], domain2_batch[0],domain1_batch[1], domain2_batch[1])
                if self.n % 10 == 0:
                    print('.', end='')
                self.n += 1

            with self.train_summary_writer.as_default():
                tf.summary.scalar('tb_gen_g_loss', self.tb_gen_g_loss.result(), step=epoch)
                tf.summary.scalar('tb_gen_f_loss', self.tb_gen_f_loss.result(), step=epoch)
                tf.summary.scalar('tb_total_cycle_loss', self.tb_total_cycle_loss.result(), step=epoch)
                tf.summary.scalar('tb_total_gen_g_loss', self.tb_total_gen_g_loss.result(), step=epoch)
                tf.summary.scalar('tb_total_gen_f_loss', self.tb_total_gen_f_loss.result(), step=epoch)
                tf.summary.scalar('tb_disc_x_loss', self.tb_disc_x_loss.result(), step=epoch)
                tf.summary.scalar('tb_disc_y_loss', self.tb_disc_y_loss.result(), step=epoch)

                tf.summary.scalar('tb_class_domain1_batch_data_F1_loss', self.tb_class_domain1_batch_data_F1_loss.result(), step=epoch)

                tf.summary.scalar('tb_ACC_domain1_batch_data_F1', self.tb_ACC_metrics_domain1_batch_data_F1.result(), step=epoch)
                tf.summary.scalar('tb_ACC_test', self.tb_ACC_metrics_domain2_batch_data_F1.result(), step=epoch)

            # Reset metrics every epoch
            self.tb_gen_g_loss.reset_states()
            self.tb_gen_f_loss.reset_states()
            self.tb_total_cycle_loss.reset_states()
            self.tb_total_gen_g_loss.reset_states()
            self.tb_total_gen_f_loss.reset_states()
            self.tb_disc_x_loss.reset_states()
            self.tb_disc_y_loss.reset_states()

            self.tb_class_domain1_batch_data_F1_loss.reset_states()

            self.tb_ACC_metrics_domain1_batch_data_F1.reset_states()
            self.tb_ACC_metrics_domain2_batch_data_F1.reset_states()

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 ckpt_save_path))
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                              time.time()-start))

    def train_procedure_visualization(self):

        if self.feature_name not in ['MRI3D']:

            for checkpoint_index in range(1,int(self.EPOCHS/5)):
                self.ckpt.restore(self.checkpoint_path + '/ckpt-' + str(checkpoint_index))

                data_domain1, data_domain2, category_domain1, category_domain2 = data_preprocessing_and_domain_selection(self.data_name,self.domain_selected[0],self.domain_selected[1],output_label_one_hot=False)

                data_domain1_to_domain2 = self.generator_g_scan(data_domain1)
                data_domain2_to_domain1 = self.generator_f_scan(data_domain2)

                data_CycleGAN_A2B = np.concatenate([data_domain1_to_domain2,data_domain2],axis=0)
                data_CycleGAN_B2A = np.concatenate([data_domain1,data_domain2_to_domain1],axis=0)

                domain_label_1 = np.ones([len(category_domain1),1]) * self.domain_selected[0]
                domain_label_2 = np.ones([len(category_domain2),1]) * self.domain_selected[1]
                domain_label = np.concatenate([domain_label_1,domain_label_2],axis=0)

                category_label = np.concatenate([category_domain1,category_domain2],axis=0)
                category_label = np.expand_dims(category_label,1)

                data_name_to_save = 'result/visualization/' + self.harmonization_mode + '_' + self.feature_name + '_ckpt' + str(checkpoint_index) + '_data_category_domain.mat'
                scio.savemat(data_name_to_save, {'data': data_CycleGAN_B2A, 'data_A2B': data_CycleGAN_A2B, 'domain_label': domain_label,
                                                 'category_label': category_label})

        if self.feature_name in ['MRI3D']:
            for checkpoint_index in range(1,int(self.EPOCHS/5)):
                self.ckpt.restore(self.checkpoint_path + '/ckpt-' + str(checkpoint_index))

                data_domain1, data_domain2, category_label_domain1_one_hot, category_label_domain2_one_hot = data_preprocessing_and_domain_selection(
                    self.s, self.domain_selected[0], self.domain_selected[1], output_label_one_hot=True,
                    feature_type=self.feature_type)
                data_domain1 = 'data/T1_data/' + pd.DataFrame(data_domain1)[0].to_numpy() + '.npy'
                data_domain2 = 'data/T1_data/' + pd.DataFrame(data_domain2)[0].to_numpy() + '.npy'
                self.domain1 = tf.data.Dataset.from_tensor_slices((data_domain1, np.float32(category_label_domain1_one_hot)))
                self.domain2 = tf.data.Dataset.from_tensor_slices((data_domain2, np.float32(category_label_domain2_one_hot)))

                index = 0
                for element in self.domain2.map(map_func=read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=1).as_numpy_iterator():
                    cycleB2A_temp = self.generator_f_scan(element[0])

                    data_name_to_save = 'result/visualization/' + self.harmonization_mode + '_' + self.feature_name + '_ckpt' + str(
                        checkpoint_index) + '_data_category_domain.mat'
                    scio.savemat(data_name_to_save, {'data': np.squeeze(cycleB2A_temp.numpy())})
                    index += 1
                    if index >= 1:
                        break
                    print(index)

    def evaluate(self):
        if self.feature_name not in ['MRI3D']:
            data_domain1, data_domain2, category_domain1, category_domain2 = data_preprocessing_and_domain_selection(self.data_name,self.domain_selected[0],self.domain_selected[1],output_label_one_hot=False)

            data_domain1_to_domain2 = self.generator_g_scan(data_domain1)
            data_domain2_to_domain1 = self.generator_f_scan(data_domain2)

            data_CycleGAN_A2B = np.concatenate([data_domain1_to_domain2,data_domain2],axis=0)
            data_CycleGAN_B2A = np.concatenate([data_domain1,data_domain2_to_domain1],axis=0)

            domain_label_1 = np.ones([len(category_domain1),1]) * self.domain_selected[0]
            domain_label_2 = np.ones([len(category_domain2),1]) * self.domain_selected[1]
            domain_label = np.concatenate([domain_label_1,domain_label_2],axis=0)

            category_label = np.concatenate([category_domain1,category_domain2],axis=0)
            category_label = np.expand_dims(category_label,1)

            data_name_to_save = 'result/'+ self.harmonization_mode + '_' + self.feature_name + '_data_category_domain.mat'
            scio.savemat(data_name_to_save, {'data': data_CycleGAN_B2A, 'data_A2B': data_CycleGAN_A2B, 'domain_label': domain_label,
                                             'category_label': category_label})
        if self.feature_name in ['MRI3D']:
            data_domain1, data_domain2, category_label_domain1_one_hot, category_label_domain2_one_hot = data_preprocessing_and_domain_selection(
                self.data_name, self.domain_selected[0], self.domain_selected[1], output_label_one_hot=True,
                feature_type=self.feature_type)
            data_domain1 = 'data/T1_data/' + pd.DataFrame(data_domain1)[0].to_numpy() + '.npy'
            data_domain2 = 'data/T1_data/' + pd.DataFrame(data_domain2)[0].to_numpy() + '.npy'
            self.domain1 = tf.data.Dataset.from_tensor_slices((data_domain1, np.float32(category_label_domain1_one_hot)))
            self.domain2 = tf.data.Dataset.from_tensor_slices((data_domain2, np.float32(category_label_domain2_one_hot)))

            temp_mask = np.zeros([121, 145, 121])
            index = 0
            for element in self.domain1.map(map_func=read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=1).as_numpy_iterator():
                cycleA2B_temp = self.generator_g_scan(element[0])
                data_name_to_save = data_domain1[index][:-4] + '_' + self.harmonization_mode + '_A2B.npy'
                temp_mask[12: -13, 8: -9, 12: -13] = np.squeeze(cycleA2B_temp.numpy())
                np.save(data_name_to_save, temp_mask)
                index += 1
                print(index)
            index = 0
            for element in self.domain2.map(map_func=read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=1).as_numpy_iterator():
                cycleB2A_temp = self.generator_f_scan(element[0])
                data_name_to_save = data_domain2[index][:-4] + '_' + self.harmonization_mode + '_B2A.npy'
                temp_mask[12: -13, 8: -9, 12: -13] = np.squeeze(cycleB2A_temp.numpy())
                np.save(data_name_to_save, temp_mask)
                index += 1
                print(index)

class MCDGAN(object):
    def __init__(self, list):

        self.feature_name = list[0]  # 'Demo'       #'MRI3D'   #'Cortical_thickness' #;
        self.lambda_discrepancy_control = list[1]
        self.harmonization_mode = 'MCDGAN'

        if self.feature_name in ['Demo', 'DemoCircle']:
            self.feature_type = 'vector'
            self.DL_model_name = 'DemoDNN'
            self.BUFFER_SIZE = 1200  # must over the sample amounts.
            self.BATCH_SIZE = 100
            self.input_vector_size = 2
            self.EPOCHS = 400
            self.domain_selected = [1, 2]
            self.lambda_cycle_loss_control = 2

        if self.feature_name in ['Cortical_thickness']:
            self.feature_type = 'vector'
            self.DL_model_name = 'DNN'
            self.BUFFER_SIZE = 10000  # must over the sample amounts.
            self.BATCH_SIZE = 100
            self.input_vector_size = 68
            self.EPOCHS = 300
            self.domain_selected = [2, 4] #
            # self.lambda_discrepancy_control = 0.05
            self.lambda_cycle_loss_control = 5

        if self.feature_name in ['MRI3D']:
            self.feature_type = 'MRI3D'
            self.DL_model_name = '3DCNN'
            self.BUFFER_SIZE = 1100
            self.BATCH_SIZE = 16
            self.input_vector_size = [96,128,96,1]
            self.EPOCHS = 100
            self.domain_selected = [2, 4] #
            self.lambda_cycle_loss_control = 5

        self.data_name = 'result/' + 'Origin_' + self.feature_name + '_data_category_domain' + '.mat'
        self.use_checkpoints = True

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'result/harmonization/logs_'+ self.feature_name +'/' + self.harmonization_mode + '_discrep_control_' + str(self.lambda_discrepancy_control) + '_domain_selected_' + str(self.domain_selected[0])+str(self.domain_selected[1])+ '_' + self.current_time
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.checkpoint_path = "result/harmonization/checkpoints_" + self.feature_name +'/' + self.harmonization_mode + '_discrep_control_' + str(self.lambda_discrepancy_control) + '_domain_selected_' + str(self.domain_selected[0])+str(self.domain_selected[1])

        data_domain1, data_domain2, category_label_domain1_one_hot, category_label_domain2_one_hot = data_preprocessing_and_domain_selection(
            self.data_name, self.domain_selected[0], self.domain_selected[1], output_label_one_hot=True,
            feature_type=self.feature_type)

        if self.feature_name not in ['MRI3D']:
            self.domain1 = tf.data.Dataset.from_tensor_slices((np.float32(data_domain1), category_label_domain1_one_hot))
            self.domain1 = self.domain1.cache().shuffle(self.BUFFER_SIZE, seed=10, reshuffle_each_iteration=True).batch(batch_size=self.BATCH_SIZE)
            self.domain2 = tf.data.Dataset.from_tensor_slices((np.float32(data_domain2), category_label_domain2_one_hot))
            self.domain2 = self.domain2.cache().shuffle(self.BUFFER_SIZE, seed=10, reshuffle_each_iteration=True).batch(batch_size=self.BATCH_SIZE)

        if self.feature_name in ['MRI3D']:
            data_domain1 = 'data/T1_data/' + pd.DataFrame(data_domain1)[0].to_numpy() + '.npy'
            data_domain2 = 'data/T1_data/' + pd.DataFrame(data_domain2)[0].to_numpy() + '.npy'
            self.domain1 = tf.data.Dataset.from_tensor_slices((data_domain1, np.float32(category_label_domain1_one_hot)))
            self.domain1 = self.domain1.map(map_func=read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(self.BUFFER_SIZE, seed=10, reshuffle_each_iteration=True).batch(batch_size=self.BATCH_SIZE)
            self.domain2 = tf.data.Dataset.from_tensor_slices((data_domain2, np.float32(category_label_domain2_one_hot)))
            self.domain2 = self.domain2.map(map_func=read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(self.BUFFER_SIZE, seed=10, reshuffle_each_iteration=True).batch(batch_size=self.BATCH_SIZE)

        self.n = 0

        print()
        print('## feature name: ', self.feature_name)
        print('Current time is: ', self.current_time)

        # def build_model(self):
        self.generator_g_scan = get_generator(generator_name=self.DL_model_name, input_vector_size=self.input_vector_size)
        self.generator_f_scan = get_generator(generator_name=self.DL_model_name, input_vector_size=self.input_vector_size)
        self.discriminator_x_scan = get_discriminator(discriminator_name=self.DL_model_name, input_vector_size=self.input_vector_size)
        self.discriminator_y_scan = get_discriminator(discriminator_name=self.DL_model_name, input_vector_size=self.input_vector_size)
        self.classifier_scan_F1 = get_classifier(classifier_name=self.DL_model_name, input_vector_size=self.input_vector_size)
        self.classifier_scan_F2 = get_classifier(classifier_name=self.DL_model_name, input_vector_size=self.input_vector_size)

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.classifier_optimizer_F1 = tf.keras.optimizers.Adam()
        self.classifier_optimizer_F2 = tf.keras.optimizers.Adam()

        self.ckpt = tf.train.Checkpoint(generator_g=self.generator_g_scan,
                                   generator_f=self.generator_f_scan,
                                   discriminator_x=self.discriminator_x_scan,
                                   discriminator_y=self.discriminator_y_scan,
                                   classifier_scan_F1=self.classifier_scan_F1,
                                   classifier_scan_F2=self.classifier_scan_F2,
                                   generator_g_optimizer=self.generator_g_optimizer,
                                   generator_f_optimizer=self.generator_f_optimizer,
                                   discriminator_x_optimizer=self.discriminator_x_optimizer,
                                   discriminator_y_optimizer=self.discriminator_y_optimizer,
                                   classifier_optimizer_F1=self.classifier_optimizer_F1,
                                   classifier_optimizer_F2=self.classifier_optimizer_F2,)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=1000)
        if self.use_checkpoints and self.ckpt_manager.latest_checkpoint:  # if a checkpoint exists and we want to use it, restore the latest checkpoint.
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest harmonization checkpoint restored==========!!')

        self.tb_gen_g_loss = tf.keras.metrics.Mean('gen_g_loss', dtype=tf.float32)
        self.tb_gen_f_loss = tf.keras.metrics.Mean('gen_f_loss', dtype=tf.float32)
        self.tb_disc_x_loss = tf.keras.metrics.Mean('disc_x_loss', dtype=tf.float32)
        self.tb_disc_y_loss = tf.keras.metrics.Mean('disc_y_loss', dtype=tf.float32)

        self.tb_class_domain1_batch_data_F1_loss = tf.keras.metrics.Mean('class_domain1_batch_data_F1_loss', dtype=tf.float32)
        self.tb_class_domain1_batch_data_F2_loss = tf.keras.metrics.Mean('class_domain1_batch_data_F2_loss', dtype=tf.float32)
        self.tb_total_cycle_loss = tf.keras.metrics.Mean('total_cycle_loss', dtype=tf.float32)
        self.tb_total_gen_g_loss = tf.keras.metrics.Mean('total_gen_g_loss', dtype=tf.float32)
        self.tb_total_gen_f_loss = tf.keras.metrics.Mean('total_gen_f_loss', dtype=tf.float32)
        self.tb_F1_F2_discrepancy_loss = tf.keras.metrics.Mean('F1_F2_discrepancy_loss', dtype=tf.float32)

        self.tb_ACC_metrics_domain1_batch_data_F1 = tf.keras.metrics.Mean('ACC_domain1_batch_data_F1',dtype=tf.float32)
        self.tb_ACC_metrics_domain1_batch_data_F2 = tf.keras.metrics.Mean('ACC_domain1_batch_data_F2',dtype=tf.float32)
        self.tb_ACC_metrics_domain2_batch_data_F1 = tf.keras.metrics.Mean('ACC_domain2_batch_data_F1', dtype=tf.float32)
        self.tb_ACC_metrics_domain2_batch_data_F2 = tf.keras.metrics.Mean('ACC_domain2_batch_data_F2', dtype=tf.float32)

        self.ACC_metrics_domain1_batch_data_F1 = tf.keras.metrics.Accuracy()
        self.ACC_metrics_domain1_batch_data_F2 = tf.keras.metrics.Accuracy()
        self.ACC_metrics_domain2_batch_data_F1 = tf.keras.metrics.Accuracy()
        self.ACC_metrics_domain2_batch_data_F2 = tf.keras.metrics.Accuracy()

    @tf.function
    def train_step1_CycleGAN(self, domain1_batch_data, domain2_batch_data,domain1_batch_label, domain2_batch_label):
        with tf.GradientTape(persistent=True) as tape:  # persistent is set to True because the tape is used more than once to calculate the gradients.
            fake_domain2_batch_data = self.generator_g_scan(domain1_batch_data, training=True)  # Generator G translates X -> Y
            cycled_domain1_batch_data = self.generator_f_scan(fake_domain2_batch_data, training=True)  # Generator F translates Y -> X
            fake_domain1_batch_data = self.generator_f_scan(domain2_batch_data, training=True)
            cycled_domain2_batch_data = self.generator_g_scan(fake_domain1_batch_data, training=True)

            same_domain1_batch_data = self.generator_f_scan(domain1_batch_data, training=True)  # same_domain1_batch_data and same_domain2_batch_data are used for identity loss.
            same_domain2_batch_data = self.generator_g_scan(domain2_batch_data, training=True)

            disc_domain1_batch_data = self.discriminator_x_scan(domain1_batch_data, training=True)
            disc_domain2_batch_data = self.discriminator_y_scan(domain2_batch_data, training=True)

            disc_fake_domain1_batch_data = self.discriminator_x_scan(fake_domain1_batch_data, training=True)
            disc_fake_domain2_batch_data = self.discriminator_y_scan(fake_domain2_batch_data, training=True)

            gen_g_loss = generator_loss(disc_fake_domain2_batch_data)
            gen_f_loss = generator_loss(disc_fake_domain1_batch_data)
            total_cycle_loss = calc_cycle_loss(domain1_batch_data, cycled_domain1_batch_data, self.lambda_cycle_loss_control) + calc_cycle_loss(domain2_batch_data,cycled_domain2_batch_data, self.lambda_cycle_loss_control)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(domain2_batch_data, same_domain2_batch_data, self.lambda_cycle_loss_control)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(domain1_batch_data, same_domain1_batch_data, self.lambda_cycle_loss_control)

            disc_x_loss = discriminator_loss(disc_domain1_batch_data, disc_fake_domain1_batch_data)
            disc_y_loss = discriminator_loss(disc_domain2_batch_data, disc_fake_domain2_batch_data)

            # write to the tensorboard
            self.tb_gen_g_loss(gen_g_loss)
            self.tb_gen_f_loss(gen_f_loss)
            self.tb_total_cycle_loss(total_cycle_loss)
            self.tb_total_gen_g_loss(total_gen_g_loss)
            self.tb_total_gen_f_loss(total_gen_f_loss)
            self.tb_disc_x_loss(disc_x_loss)
            self.tb_disc_y_loss(disc_y_loss)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g_scan.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f_scan.trainable_variables)

        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, self.generator_g_scan.trainable_variables))
        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, self.generator_f_scan.trainable_variables))

        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x_scan.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y_scan.trainable_variables)
        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, self.discriminator_x_scan.trainable_variables))
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, self.discriminator_y_scan.trainable_variables))

    @tf.function
    def train_step2_max_discrepancy(self, domain1_batch_data, domain2_batch_data, domain1_batch_label, domain2_batch_label):
        with tf.GradientTape(persistent=True) as tape:  # persistent is set to True because the tape is used more than once to calculate the gradients.
            fake_domain1_batch_data = self.generator_f_scan(domain2_batch_data, training=True)

            classified_domain1_batch_data_F1 = self.classifier_scan_F1(domain1_batch_data, training=True)
            classified_domain1_batch_data_F2 = self.classifier_scan_F2(domain1_batch_data, training=True)
            classified_fake_domain1_batch_data_F1 = self.classifier_scan_F1(fake_domain1_batch_data, training=True)
            classified_fake_domain1_batch_data_F2 = self.classifier_scan_F2(fake_domain1_batch_data, training=True)

            class_domain1_batch_data_F1_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(domain1_batch_label, classified_domain1_batch_data_F1))
            class_domain1_batch_data_F2_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(domain1_batch_label, classified_domain1_batch_data_F2))
            F1_F2_discrepancy_loss = tf.reduce_mean(tf.abs(tf.subtract(classified_fake_domain1_batch_data_F1, classified_fake_domain1_batch_data_F2)))

            ## This section is to make sure the loss is not below zero.

            total_class_max_discrepancy_loss_1 = class_domain1_batch_data_F1_loss - self.lambda_discrepancy_control * F1_F2_discrepancy_loss
            total_class_max_discrepancy_loss_2 = class_domain1_batch_data_F2_loss - self.lambda_discrepancy_control * F1_F2_discrepancy_loss


            self.ACC_metrics_domain1_batch_data_F1.update_state(tf.math.argmax(domain1_batch_label, 1),
                                                                tf.math.argmax(classified_domain1_batch_data_F1, 1))
            self.ACC_metrics_domain1_batch_data_F2.update_state(tf.math.argmax(domain1_batch_label, 1),
                                                                tf.math.argmax(classified_domain1_batch_data_F2, 1))
            self.ACC_metrics_domain2_batch_data_F1.update_state(tf.math.argmax(domain2_batch_label, 1),
                                                                tf.math.argmax(classified_fake_domain1_batch_data_F1, 1))
            self.ACC_metrics_domain2_batch_data_F2.update_state(tf.math.argmax(domain2_batch_label, 1),
                                                                tf.math.argmax(classified_fake_domain1_batch_data_F1, 1))
            # write to the tensorboard
            self.tb_class_domain1_batch_data_F1_loss(class_domain1_batch_data_F1_loss)
            self.tb_class_domain1_batch_data_F2_loss(class_domain1_batch_data_F2_loss)
            self.tb_F1_F2_discrepancy_loss(F1_F2_discrepancy_loss)

            self.tb_ACC_metrics_domain1_batch_data_F1(self.ACC_metrics_domain1_batch_data_F1.result())
            self.tb_ACC_metrics_domain1_batch_data_F2(self.ACC_metrics_domain1_batch_data_F2.result())
            self.tb_ACC_metrics_domain2_batch_data_F1(self.ACC_metrics_domain2_batch_data_F1.result())
            self.tb_ACC_metrics_domain2_batch_data_F2(self.ACC_metrics_domain2_batch_data_F1.result())

        classifier_gradients_F1 = tape.gradient(total_class_max_discrepancy_loss_1, self.classifier_scan_F1.trainable_variables)
        classifier_gradients_F2 = tape.gradient(total_class_max_discrepancy_loss_2, self.classifier_scan_F2.trainable_variables)

        self.classifier_optimizer_F1.apply_gradients(zip(classifier_gradients_F1, self.classifier_scan_F1.trainable_variables))
        self.classifier_optimizer_F2.apply_gradients(zip(classifier_gradients_F2, self.classifier_scan_F2.trainable_variables))

    @tf.function
    def train_step3(self, domain1_batch_data, domain2_batch_data,domain1_batch_label, domain2_batch_label):
        with tf.GradientTape(persistent=True) as tape:  # persistent is set to True because the tape is used more than once to calculate the gradients.
            fake_domain1_batch_data = self.generator_f_scan(domain2_batch_data, training=True)
            disc_fake_domain1_batch_data = self.discriminator_x_scan(fake_domain1_batch_data, training=True)

            classified_fake_domain1_batch_data_F1 = self.classifier_scan_F1(fake_domain1_batch_data, training=True)
            classified_fake_domain1_batch_data_F2 = self.classifier_scan_F2(fake_domain1_batch_data, training=True)

            gen_f_loss = generator_loss(disc_fake_domain1_batch_data)

            F1_F2_discrepancy_loss = tf.reduce_mean(tf.abs(tf.subtract(classified_fake_domain1_batch_data_F1, classified_fake_domain1_batch_data_F2)))

            total_gen_f_loss_pluse_F1_F2_discrepancy_loss = self.lambda_discrepancy_control * F1_F2_discrepancy_loss + gen_f_loss

        # Calculate the gradients for generator and discriminator
        generator_f_gradients = tape.gradient(total_gen_f_loss_pluse_F1_F2_discrepancy_loss, self.generator_f_scan.trainable_variables)
        # Apply the gradients to the optimizer
        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, self.generator_f_scan.trainable_variables))

    def train(self):
        for epoch in range(self.EPOCHS):
            start = time.time()
            print('Epoch: ', epoch)

            for domain1_batch, domain2_batch in tf.data.Dataset.zip((self.domain1, self.domain2)):
                self.train_step1_CycleGAN(domain1_batch[0], domain2_batch[0],domain1_batch[1], domain2_batch[1])
                if epoch > 20:
                    self.train_step2_max_discrepancy(domain1_batch[0], domain2_batch[0],domain1_batch[1], domain2_batch[1])
                if epoch > 40:
                    self.train_step3(domain1_batch[0], domain2_batch[0],domain1_batch[1], domain2_batch[1])

                if self.n % 10 == 0:
                    print('.', end='')
                self.n += 1

            with self.train_summary_writer.as_default():
                tf.summary.scalar('tb_gen_g_loss', self.tb_gen_g_loss.result(), step=epoch)
                tf.summary.scalar('tb_gen_f_loss', self.tb_gen_f_loss.result(), step=epoch)
                tf.summary.scalar('tb_total_cycle_loss', self.tb_total_cycle_loss.result(), step=epoch)
                tf.summary.scalar('tb_total_gen_g_loss', self.tb_total_gen_g_loss.result(), step=epoch)
                tf.summary.scalar('tb_total_gen_f_loss', self.tb_total_gen_f_loss.result(), step=epoch)
                tf.summary.scalar('tb_disc_x_loss', self.tb_disc_x_loss.result(), step=epoch)
                tf.summary.scalar('tb_disc_y_loss', self.tb_disc_y_loss.result(), step=epoch)

                tf.summary.scalar('tb_class_domain1_batch_data_F1_loss', self.tb_class_domain1_batch_data_F1_loss.result(), step=epoch)

                tf.summary.scalar('tb_ACC_domain1_batch_data_F1', self.tb_ACC_metrics_domain1_batch_data_F1.result(), step=epoch)
                tf.summary.scalar('tb_ACC_domain1_batch_data_F2', self.tb_ACC_metrics_domain1_batch_data_F2.result(), step=epoch)
                tf.summary.scalar('tb_ACC_domain2_batch_data_F1', self.tb_ACC_metrics_domain2_batch_data_F1.result(), step=epoch)
                tf.summary.scalar('tb_ACC_domain2_batch_data_F2', self.tb_ACC_metrics_domain2_batch_data_F2.result(), step=epoch)

                tf.summary.scalar('tb_F1_F2_discrepancy_loss', self.tb_F1_F2_discrepancy_loss.result(), step=epoch)

            # Reset metrics every epoch
            self.tb_gen_g_loss.reset_states()
            self.tb_gen_f_loss.reset_states()
            self.tb_total_cycle_loss.reset_states()
            self.tb_total_gen_g_loss.reset_states()
            self.tb_total_gen_f_loss.reset_states()
            self.tb_disc_x_loss.reset_states()
            self.tb_disc_y_loss.reset_states()
            self.tb_F1_F2_discrepancy_loss.reset_states()

            self.tb_class_domain1_batch_data_F1_loss.reset_states()
            self.tb_ACC_metrics_domain1_batch_data_F1.reset_states()
            self.tb_ACC_metrics_domain1_batch_data_F2.reset_states()

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

    def train_procedure_visualization(self):
        if self.feature_name not in ['MRI3D']:

            for checkpoint_index in range(1,int(self.EPOCHS/5)):
                self.ckpt.restore(self.checkpoint_path + '/ckpt-' + str(checkpoint_index))

                data_domain1, data_domain2, category_domain1, category_domain2 = data_preprocessing_and_domain_selection(self.data_name,self.domain_selected[0],self.domain_selected[1],output_label_one_hot=False)

                data_domain1_to_domain2 = self.generator_g_scan(data_domain1)
                data_domain2_to_domain1 = self.generator_f_scan(data_domain2)

                data_CycleGAN_A2B = np.concatenate([data_domain1_to_domain2,data_domain2],axis=0)
                data_CycleGAN_B2A = np.concatenate([data_domain1,data_domain2_to_domain1],axis=0)

                domain_label_1 = np.ones([len(category_domain1),1]) * self.domain_selected[0]
                domain_label_2 = np.ones([len(category_domain2),1]) * self.domain_selected[1]
                domain_label = np.concatenate([domain_label_1,domain_label_2],axis=0)

                category_label = np.concatenate([category_domain1,category_domain2],axis=0)
                category_label = np.expand_dims(category_label,1)

                data_name_to_save = 'result/visualization/'+ self.harmonization_mode + '_' + self.feature_name + '_discrep_control_' + str(self.lambda_discrepancy_control) + '_ckpt' + str(checkpoint_index) + '_data_category_domain.mat'
                scio.savemat(data_name_to_save, {'data': data_CycleGAN_B2A, 'data_A2B': data_CycleGAN_A2B, 'domain_label': domain_label,
                                                 'category_label': category_label})

        if self.feature_name in ['MRI3D']:
            for checkpoint_index in range(1,int(self.EPOCHS/5)):
                self.ckpt.restore(self.checkpoint_path + '/ckpt-' + str(checkpoint_index))

                data_domain1, data_domain2, category_label_domain1_one_hot, category_label_domain2_one_hot = data_preprocessing_and_domain_selection(
                    self.data_name, self.domain_selected[0], self.domain_selected[1], output_label_one_hot=True,
                    feature_type=self.feature_type)

                data_domain1 = 'data/T1_data/' + pd.DataFrame(data_domain1)[0].to_numpy() + '.npy'
                data_domain2 = 'data/T1_data/' + pd.DataFrame(data_domain2)[0].to_numpy() + '.npy'
                self.domain1 = tf.data.Dataset.from_tensor_slices((data_domain1, np.float32(category_label_domain1_one_hot)))
                self.domain2 = tf.data.Dataset.from_tensor_slices((data_domain2, np.float32(category_label_domain2_one_hot)))

                index = 0
                for element in self.domain2.map(map_func=read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=1).as_numpy_iterator():
                    cycleB2A_temp = self.generator_f_scan(element[0])

                    data_name_to_save = 'result/visualization/' + self.harmonization_mode + '_' + self.feature_name + '_ckpt' + str(
                        checkpoint_index) + '_data_category_domain.mat'
                    scio.savemat(data_name_to_save, {'data': np.squeeze(cycleB2A_temp.numpy())})
                    index += 1
                    if index >= 1:
                        break
                    print(index)

    def evaluate(self):

        if self.feature_name not in ['MRI3D']:
            data_domain1, data_domain2, category_domain1, category_domain2 = data_preprocessing_and_domain_selection(self.data_name,self.domain_selected[0],self.domain_selected[1],output_label_one_hot=False)

            data_domain1_to_domain2 = self.generator_g_scan(data_domain1)
            data_domain2_to_domain1 = self.generator_f_scan(data_domain2)

            data_CycleGAN_A2B = np.concatenate([data_domain1_to_domain2,data_domain2],axis=0)
            data_CycleGAN_B2A = np.concatenate([data_domain1,data_domain2_to_domain1],axis=0)

            domain_label_1 = np.ones([len(category_domain1),1]) * self.domain_selected[0]
            domain_label_2 = np.ones([len(category_domain2),1]) * self.domain_selected[1]
            domain_label = np.concatenate([domain_label_1,domain_label_2],axis=0)

            category_label = np.concatenate([category_domain1,category_domain2],axis=0)
            category_label = np.expand_dims(category_label,1)

            data_name_to_save = 'result/'+ self.harmonization_mode + '_' + self.feature_name + '_discrep_control_' + str(self.lambda_discrepancy_control) + '_data_category_domain.mat'
            scio.savemat(data_name_to_save, {'data': data_CycleGAN_B2A, 'data_A2B': data_CycleGAN_A2B, 'domain_label': domain_label,
                                             'category_label': category_label})

        if self.feature_name in ['MRI3D']:
            data_domain1, data_domain2, category_label_domain1_one_hot, category_label_domain2_one_hot = data_preprocessing_and_domain_selection(
                self.data_name, self.domain_selected[0], self.domain_selected[1], output_label_one_hot=True,
                feature_type=self.feature_type)
            data_domain1 = 'data/T1_data/' + pd.DataFrame(data_domain1)[0].to_numpy() + '.npy'
            data_domain2 = 'data/T1_data/' + pd.DataFrame(data_domain2)[0].to_numpy() + '.npy'
            self.domain1 = tf.data.Dataset.from_tensor_slices((data_domain1, np.float32(category_label_domain1_one_hot)))
            self.domain2 = tf.data.Dataset.from_tensor_slices((data_domain2, np.float32(category_label_domain2_one_hot)))

            temp_mask = np.zeros([121,145,121])

            index = 0
            for element in self.domain2.map(map_func=read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=1).as_numpy_iterator():
                cycleB2A_temp = self.generator_f_scan(element[0])
                data_name_to_save = data_domain2[index][:-4] + '_' + self.harmonization_mode + '_B2A_' + str(self.lambda_discrepancy_control) + '.npy'
                temp_mask[12: -13, 8: -9, 12: -13] = np.squeeze(cycleB2A_temp.numpy())
                np.save(data_name_to_save, temp_mask)
                index += 1
                print(index)

def read_npy_file(filename, label):
    '''
    Example: data = tf.py_function(read_npy_file,[npy_file_names[0]],[tf.float32])

    :param filename:
    :return:
    '''
    data = np.load(filename.numpy().decode())
    data = data[12: -13, 8: -9, 12: -13]
    data = np.expand_dims(data, axis=3)
    return data.astype(np.float32), label

def read_image(filename, label):
    '''
    Example: read_image(npy_file_names[0])

    :param filename:
    :return:
    '''
    image_, label = tf.py_function(read_npy_file, [filename, label], [tf.float32, tf.float32])
    print(tf.shape(image_))
    return image_, label
