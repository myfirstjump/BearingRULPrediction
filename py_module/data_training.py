import time
import os
import random
import itertools

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import model_selection

from py_module.neural_design import NeuralCalculation, LossDesign
from py_module.plot_module import PlotDesign
from py_module.config import Configuration
from py_module.learning_definition import LearningDefinition

class DataTraining(object):

    def __init__(self):
        self.neural_obj = NeuralCalculation()
        self.loss_obj = LossDesign()
        self.plot_obj = PlotDesign()
        self.config_obj = Configuration()
        self.learing_def_obj = LearningDefinition()

    def sys_show_execution_time(method):
        def time_record(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            execution_time = np.round(end_time - start_time, 3)
            print('Running function:', method.__name__, ' cost time:', execution_time, 'seconds.')
            return result
        return time_record

    ### custom loss
    def custom_loss_function(self, y_true, y_pred):

        # print("y_true:", y_true)
        diff_vec = y_true - y_pred
        def each_loss_apply(d):
            diff_value = tf.cond(d<0, true_fn=lambda: tf.exp(-d/9)-1, false_fn=lambda: tf.exp(d/13)-1)
            # diff_value = tf.cond(d<0, true_fn=lambda: -5*d, false_fn=lambda: tf.exp(d/8)-1)
            return diff_value
        diff_vec = tf.map_fn(each_loss_apply, diff_vec)

        return tf.reduce_mean(diff_vec)


    # def grid_search_with_cross_validation(self, hyperparameters, data, cv_fold, preprocessing_function, training_function, scoring_metric):

    #     for element in itertools.product(*hyperparameters):
    #         print(element)

    @sys_show_execution_time
    def training_2008_PHM_Engine_data(self, data, epochs, load_model = True):
        
        # Split train/valid
        train_units, valid_units = model_selection.train_test_split([i+1 for i in range(self.config_obj.train_engine_number)], test_size=0.2)

        # Produce generator for differenet engine unit
        def yield_unit_data(data, train_valid_units, epochs=epochs):
            cnt = 0
            while cnt < epochs:
                which_unit = random.choice(train_valid_units)
                unit_data = data[data['unit'] == which_unit]
                cnt += 1
                yield which_unit, unit_data
        train_data_generator = yield_unit_data(data, train_units, epochs)

        ### mse loss
        # def custom_loss_function(y_true, y_pred):
        #     squared_difference = tf.square(y_true - y_pred)
        #     return tf.reduce_mean(squared_difference, axis=-1)

        checkpoint_path = self.config_obj.checkpoint_path
        h5_path = self.config_obj.keras_model_path
        new_h5_path = self.config_obj.keras_updated_model_path
        my_history = {'train_loss':[], 'valid_loss':[]}

        ### load model or not
        if load_model:
            if os.path.isfile(h5_path):
                model = keras.models.load_model(h5_path)
            else:
                model = self.model_design('RNN')
            model.compile(optimizer = keras.optimizers.RMSprop(), loss = self.custom_loss_function, metrics = ['mse'])
        else:
            model = self.model_design('RNN')
            model.compile(optimizer = keras.optimizers.RMSprop(), loss = self.custom_loss_function, metrics = ['mse'])

        training_cnt = 1

        for train_unit_num, train_data in train_data_generator:

            print("======================================= Training Epoch {} =====================================".format(training_cnt))

            valid_unit_num, valid_data = [(valid_unit_num, valid_data) for (valid_unit_num, valid_data) in yield_unit_data(data, valid_units, 1)][0]

            train_data = self.learing_def_obj.learning_define_2008_PHM_Engine_data(train_data)
            valid_data = self.learing_def_obj.learning_define_2008_PHM_Engine_data(valid_data)
            print("以引擎 unit: {} 做為training data.".format(train_unit_num))
            print("以引擎 unit: {} 做為validation data.".format(valid_unit_num))

            train_x = train_data.values[:,:-1]
            train_y = train_data.values[:, -1]
            valid_x = valid_data.values[:,:-1]
            valid_y = valid_data.values[:, -1]

            # Reshape
            train_x = train_x.reshape((train_x.shape[0], self.config_obj.previous_p_times + 1, self.config_obj.features_num))
            valid_x = valid_x.reshape((valid_x.shape[0], self.config_obj.previous_p_times + 1, self.config_obj.features_num))

            # 
            model.reset_states()
            model, history = self.RNN_model_training(model, (train_x, train_y, valid_x, valid_y), checkpoint_path)

            print(history.history['loss'], history.history['val_loss'])
            my_history['train_loss'].append(history.history['loss'][0])
            my_history['valid_loss'].append(history.history['val_loss'][0])
            training_cnt += 1
        model.save(new_h5_path)
        print("Save Model at {}".format(new_h5_path))
        return my_history


        
    def model_design(self, model_name, graph_output_dir=None):
        
        if model_name == 'RNN':

            model = keras.models.Sequential()

            ### normal
            # model.add(keras.layers.GRU(64, input_shape=(self.config_obj.previous_p_times+1, self.config_obj.features_num), return_sequences=True))
            # model.add(keras.layers.GRU(64, return_sequences=True))
            # model.add(keras.layers.GRU(32))
            # model.add(keras.layers.Dense(32))
            # model.add(keras.layers.Dense(16))

            ### overfit if
            dropout_rate = 0.5
            model.add(keras.layers.GRU(128, input_shape=(self.config_obj.previous_p_times+1, self.config_obj.features_num), return_sequences=True))
            model.add(keras.layers.Dropout(rate = dropout_rate))
            model.add(keras.layers.GRU(64, return_sequences=True))
            model.add(keras.layers.Dropout(rate = dropout_rate))
            model.add(keras.layers.GRU(64, return_sequences=True))
            model.add(keras.layers.Dropout(rate = dropout_rate))
            model.add(keras.layers.GRU(32))
            model.add(keras.layers.Dropout(rate = dropout_rate))
            model.add(keras.layers.Dense(32))
            model.add(keras.layers.Dropout(rate = dropout_rate))
            model.add(keras.layers.Dense(16))
            model.add(keras.layers.Dropout(rate = dropout_rate))


            model.add(keras.layers.Dense(1))

        if model_name == 'Autoencoder':
            
            origin_dim = hyperparameters['origin_dim']
            encoding_dim = hyperparameters['encoding_dim']


            input_img = Input(shape=(origin_dim,))
            encoded = Dense(12, activation='relu')(input_img)
            encoded = Dense(encoding_dim, activation='relu')(encoded)

            decoded = Dense(12, activation='relu')(encoded)
            decoded = Dense(origin_dim, activation='sigmoid')(decoded)
            

        if model_name == 'DNN':
            model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dense(256, activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
        
        if model_name == 'CNN':
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=64, input_shape=(28,28,1), kernel_size=(3,3), strides=1, padding='valid', activation=tf.nn.relu),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='valid', activation=tf.nn.relu),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])

        if model_name == 'GAN':

            # Two kinds of modeling method:
            # 1. Separately define G_net and D_net    <-   We use this.
            # 2. Define the hidden layer involve G and D. Fix the G weigths when training D, and vice versa.
            
            seed = hyperparameters[' ']
            batch_size = hyperparameters['batch_size']
            X_dim = hyperparameters['X_dim']
            z_dim = hyperparameters['z_dim']
            h_dim = hyperparameters['h_dim']
            lam = hyperparameters['lam']
            n_disc = hyperparameters['n_disc']
            lr = hyperparameters['lr']

            tf.set_random_seed(seed)
            np.random.seed(seed)

            tf.reset_default_graph()

            X = tf.placeholder(tf.float32, shape=[None, X_dim])
            X_target = tf.placeholder(tf.float32, shape=[None, X_dim])
            z = tf.placeholder(tf.float32, shape=[None, z_dim])

            G_sample, G_var = self.neural_obj.generator(z) # 由 m = 32 vectors
            D_real_logits, D_var = self.neural_obj.discriminator(X, spectral_normed=False)
            D_fake_logits, _ = self.neural_obj.discriminator(G_sample, spectral_normed=False, reuse=True)

            D_loss, G_loss = self.loss_obj.gan_loss(D_real_logits, D_fake_logits, gan_type='GAN', relativistic=False)
            D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)).minimize(D_loss, var_list=D_var)
            G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)).minimize(G_loss, var_list=G_var)

            # z search
            z_optimizer = tf.train.AdamOptimizer(0.0001)
            z_r = tf.get_variable('z_update', [batch_size, z_dim], tf.float32)
            G_z_r, _ = self.neural_obj.generator(z_r, reuse=True)

            z_r_loss = tf.reduce_mean(tf.abs(tf.reshape(X_target, [-1, 28, 28, 1]) - G_z_r))
            z_r_optim = z_optimizer.minimize(z_r_loss, var_list=[z_r])

            sess = tf.Session()

            tensorboard_output = graph_output_dir + 'gan_graphs'
            writer = tf.summary.FileWriter(tensorboard_output, sess.graph)
            sess.run(tf.global_variables_initializer())

            model = {}
            model['sess'] = sess
            model['D_solver'] = D_solver
            model['D_loss'] = D_loss
            model['G_solver'] = G_solver
            model['G_loss'] = G_loss
            model['G_sample'] = G_sample
            model['X'] = X
            model['z'] = z

        return model

    @sys_show_execution_time
    def RNN_model_training(self, model, data, checkpoint_path):
        
        train_x, train_y, valid_x, valid_y = data
        
        # earlystopping = keras.callbacks.Earlystopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
        # cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        # callbacks= [earlystopping, cp_callback]
        history = model.fit(train_x, train_y, epochs=1, batch_size=16, validation_data=(valid_x, valid_y), verbose=2, shuffle=False)

        return model, history

    @sys_show_execution_time
    def model_training(self, model, data):
        
        training_images, training_labels = data
        model.compile(optimizer = 'adam',#optimizer = tf.compat.v1.train.AdamOptimizer(),
                      loss =  'sparse_categorical_crossentropy',
                      metrics = ['accuracy'])
        callbacks = CallBack()
        model.fit(training_images, training_labels, epochs=15, callbacks=[callbacks])

        return model

    @sys_show_execution_time
    def gan_model_training(self, data, output_dir, model, hyperparameters=None):

        batch_size = hyperparameters['batch_size']
        X_dim = hyperparameters['X_dim']
        z_dim = hyperparameters['z_dim']
        h_dim = hyperparameters['h_dim']
        lam = hyperparameters['lam']
        n_disc = hyperparameters['n_disc']
        lr = hyperparameters['lr']

        sess = model['sess']
        D_solver = model['D_solver']
        D_loss = model['D_loss']
        G_solver = model['G_solver']
        G_loss = model['G_loss']
        G_sample = model['G_sample']
        X = model['X']
        z = model['z']

        start_time = time.time()
        for it in range(300000):
            for _ in range(n_disc):

                # First, fix generator G, and update discriminator D.
                X_mb, _ = data.train.next_batch(batch_size)

                _, D_loss_curr = sess.run(
                    [D_solver, D_loss],
                    feed_dict={X: X_mb, z: self.neural_obj.sample_z(batch_size, z_dim)}
                )
            
            # Second, fix discriminator, and update generator G.
            X_mb, _ = data.train.next_batch(batch_size)
            _, G_loss_curr = sess.run(
                [G_solver, G_loss],
                feed_dict={X: X_mb, z: self.neural_obj.sample_z(batch_size, z_dim)}
            )

            if it % 10000 == 0:
                print('Iter: {}; Cost Time: {:.4}; D loss: {:.4}; G_loss: {:.4}'.format(it, time.time() - start_time, D_loss_curr, G_loss_curr))
                
                samples = sess.run(G_sample, feed_dict={z: self.neural_obj.sample_z(16, z_dim)})
                fig = self.plot_obj.plot(samples)

                dest_path = output_dir + 'gan_output/'
                self.plot_obj.plot_saving(dest_path=dest_path, filename='gan_generator_{}_{}'.format('mnist', it), suffix='png')

class CallBack(tf.keras.callbacks.Callback):

    # Each epoch end, will call the method on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.98):
            print('Reached enough accuracy so stop training...')
            self.model.stop_training = True
