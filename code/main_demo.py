# -*- coding: utf-8 -*-
'''
2022-05-01 Weizheng Yan conanywz@gmail.com
ersion: v0.1
'''

import Harmony_toolkit
import shutil
import os
import tensorflow as tf


def LetUsHarmony(feature_name, harmonization_mode, harmonization_retrain, lambda_discrepancy_control=1):
    '''

    Parameters
    ----------
    feature_name: 'Demo','DemoCircle','Cortical_thickness','MRI3D'
    harmonization_mode: Options: 'Origin', 'ComBat', 'cycleGAN', 'MCDGAN'
    harmonization_retrain: 1: retrain the harmony, 0: otherwise
    lambda_discrepancy_control:

    Returns
    -------

    '''

    print('Starting harmony')
    print(harmonization_mode)

    '''
        If you want to retrain the harmony mode, remove the previous checkpoints.
    '''

    if harmonization_mode in ['CycleGAN']:
        if feature_name in ['Demo','DemoCircle']:
            harmonization_checkpoint_dir_to_remove_if_train = 'result/harmonization/checkpoints_' + feature_name + '/' + harmonization_mode + '_domain_selected_12'
        else:
            harmonization_checkpoint_dir_to_remove_if_train = 'result/harmonization/checkpoints_' + feature_name + '/' + harmonization_mode + '_domain_selected_24'
    if harmonization_mode in ['MCDGAN']: # MCDGAN requires "discrepancy control".
        if feature_name in ['Demo', 'DemoCircle']:
            harmonization_checkpoint_dir_to_remove_if_train = 'result/harmonization/checkpoints_' + feature_name + '/' + harmonization_mode + '_discrep_control_' + str(lambda_discrepancy_control) + '_domain_selected_12'
        else:
            harmonization_checkpoint_dir_to_remove_if_train = 'result/harmonization/checkpoints_' + feature_name + '/' + harmonization_mode + '_discrep_control_' + str(lambda_discrepancy_control) + '_domain_selected_24'

    '''
    Harmonization
    '''
    if harmonization_mode in ['ComBat']:
        My_ComBat = Harmony_toolkit.ComBat(feature_name)

        if harmonization_retrain:
            print('Retraining ComBat......')
            My_ComBat = Harmony_toolkit.ComBat(feature_name)
            My_ComBat.apply_combat()
            My_ComBat.save_combat_to_each_subject()

    if harmonization_mode in ['CycleGAN']:
        if harmonization_retrain:
            print('Removing CycleGAN checkpoints......')
            if os.path.exists(harmonization_checkpoint_dir_to_remove_if_train):
                shutil.rmtree(harmonization_checkpoint_dir_to_remove_if_train)

        My_CycleGAN = Harmony_toolkit.CycleGAN(feature_name)
        if harmonization_retrain:
            print('Retraining the CycleGAN harmonization......')
            My_CycleGAN.train()
            # My_CycleGAN.train_procedure_visualization()
            My_CycleGAN.evaluate()

    if harmonization_mode in ['MCDGAN']:
        if harmonization_retrain:
            print('Removing MCDGAN checkpoints......')
            if os.path.exists(harmonization_checkpoint_dir_to_remove_if_train):
                shutil.rmtree(harmonization_checkpoint_dir_to_remove_if_train)

        My_MCDGAN = Harmony_toolkit.MCDGAN([feature_name, lambda_discrepancy_control])
        if harmonization_retrain:
            print('Retraining the MCDGAN harmonization......')
            My_MCDGAN.train()
            # My_MCDGAN.train_procedure_visualization()
            My_MCDGAN.evaluate()

tf.compat.v1.app.flags.DEFINE_string('harmony_mode', 'CycleGAN', 'Harmonization method to select: can be ComBat, CycleGAN or MCDGAN')
tf.compat.v1.app.flags.DEFINE_string('feature_name', 'Demo', 'Feature_name_to_select: can be Demo, MRI3D, Average_T1_intensity')
tf.compat.v1.app.flags.DEFINE_integer('harmonization_retrain', 1, 'whether to retrain the harmonization')
tf.compat.v1.app.flags.DEFINE_float('lambda_discrepancy_control', 1.0, 'Lambda_discrepancy_control')

FLAGS = tf.compat.v1.app.flags.FLAGS

def main(_):

    harmonization_mode = FLAGS.harmony_mode
    feature_name = FLAGS.feature_name
    harmonization_retrain = FLAGS.harmonization_retrain
    lambda_discrepancy_control = FLAGS.lambda_discrepancy_control

    print('harmonization_mode: ', harmonization_mode)
    print('feature_name: ', feature_name)
    print('harmonization_retrain: ', harmonization_retrain)
    print('Lambda_discrepancy_control: ', lambda_discrepancy_control)

    LetUsHarmony(feature_name, harmonization_mode, harmonization_retrain, lambda_discrepancy_control)

if __name__=='__main__':
    tf.compat.v1.app.run()

    # Step1: Copy and paste the following and run in the Linux shell for harmonization.

    # python main_demo.py -harmony_mode=ComBat -feature_name=Demo --harmonization_retrain=1
    # python main_demo.py -harmony_mode=CycleGAN -feature_name=Demo --harmonization_retrain=1
    # python main_demo.py -harmony_mode=MCDGAN -feature_name=Demo --harmonization_retrain=1  --lambda_discrepancy_control=3.2

    # Step2: Visualization
    # python demo_visualize.
