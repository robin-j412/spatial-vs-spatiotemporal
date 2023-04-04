import argparse

from data import PS, EX_ZONE, EXPE, load_train_val_test_ex
from models import make_transformer_patch, make_transformer_spat, make_transformer_temp_only, efficient_net_pretrain, baseline_cnn
import tensorflow as tf
from datetime import datetime
import os
import pandas as pd
import numpy as np


gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

class LRCB(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        logs['learning_rate'] = self.model.optimizer._decayed_lr('float32').numpy()
        return logs


class ValCB(tf.keras.callbacks.Callback):

    def __init__(self, monitor='val_MAE'):
        super(ValCB, self).__init__()

        self.monitor = monitor
        self.best_val_mean_absolute_error = 1e19 if 'r2' not in self.monitor else -1e19
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):

        if 'r2' in self.monitor:
            cond = self.best_val_mean_absolute_error <= logs[self.monitor]
        else:
            cond = self.best_val_mean_absolute_error >= logs[self.monitor]

        if cond:
            self.best_val_mean_absolute_error = logs[self.monitor]
            self.best_epoch = epoch

        logs['best_' + self.monitor] = self.best_val_mean_absolute_error
        logs['best_epoch_' + self.monitor] = self.best_epoch


def callbacks(expe_name, test_fold, year=None):
    learning_rate_cb = LRCB()

    val_mae_cb = ValCB(monitor='val_mae')
    val_loss_cb = ValCB(monitor='val_loss'),

    if year is None:
        logs = EXPE + f'/zone-{EX_ZONE[0]}_ps-{PS}/' + \
               f'models/{expe_name}/logs/{test_fold}/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        logs = EXPE + f'/zone-{EX_ZONE[0]}_ps-{PS}/' + \
               f'models/{expe_name}/{year}/logs/{test_fold}/' + datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(logs, exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch=(1, 30)),

    save_best_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=logs + '/model-best', save_weights_only=True,
                                                            monitor='val_loss',
                                                            save_best_only=True, verbose=1)

    save_last_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=logs + '/model-last', save_weights_only=True,
                                                            monitor='val_loss',
                                                            save_best_only=False, verbose=1)

    callbacks_list = [learning_rate_cb, val_loss_cb, val_mae_cb, tensorboard_cb, save_best_model_cb, save_last_model_cb]

    return callbacks_list, logs


def run_baseline(train, val, test, ntl_type, epochs, batch_size, learning_rate, num_heads, key_dim, depth):

    (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_ex, y_ex) = load_train_val_test_ex(train=train, val=val,
                                                                                                test=test,
                                                                                                ntl_type=ntl_type)
    input_shape = X_train.shape[1:]
    output_shape = (y_train.shape[1],)
    model = make_transformer_patch(input_shape, out_shape=output_shape,
                                   num_heads=num_heads, key_dim=key_dim, depth=depth)

    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                         decay_steps=(7000 // batch_size) * 2,
                                                                         decay_rate=1., staircase=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate_decay),
                  metrics=['mae'])

    callback_list, logs = callbacks('baseline', test_fold=test)

    h = model.fit(X_train,
                  y_train,
                  epochs=epochs,
                  callbacks=callback_list,
                  batch_size=batch_size,
                  validation_data=(X_val, y_val))

    model.load_weights(logs + '/model-best')

    pd.DataFrame(h.history).to_csv(logs + '/history.csv')

    predictions = model.predict(X_test, verbose=1)
    np.save(logs + '/predictions.npy', predictions)

    predictions = model.predict(X_ex)
    np.save(logs + '/ex_zone_pred.npy', predictions)

    del X_train, y_train, X_test, y_test


def run_temp_only(train, val, test, ntl_type, epochs, batch_size, learning_rate, num_heads, key_dim, depth):

    (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_ex, y_ex) = load_train_val_test_ex(train=train, val=val,
                                                                                                test=test,
                                                                                                ntl_type=ntl_type)
    input_shape = X_train.shape[1:]
    output_shape = (y_train.shape[1],)
    model = make_transformer_temp_only(input_shape, out_shape=output_shape,
                                       num_heads=num_heads, key_dim=key_dim, depth=depth)

    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                         decay_steps=(7000 // batch_size) * 2,
                                                                         decay_rate=1., staircase=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate_decay),
                  metrics=['mae'])

    callback_list, logs = callbacks('temp_only', test_fold=test)

    h = model.fit(X_train,
                  y_train,
                  epochs=epochs,
                  callbacks=callback_list,
                  batch_size=batch_size,
                  validation_data=(X_val, y_val))

    model.load_weights(logs + '/model-best')

    pd.DataFrame(h.history).to_csv(logs + '/history.csv')

    predictions = model.predict(X_test, verbose=1)
    np.save(logs + '/predictions.npy', predictions)

    predictions = model.predict(X_ex)
    np.save(logs + '/ex_zone_pred.npy', predictions)

    del X_train, y_train, X_test, y_test


def run_spat(train, val, test, ntl_type, epochs, batch_size, learning_rate, num_heads, key_dim, depth):

    (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_ex, y_ex) = load_train_val_test_ex(train=train, val=val,
                                                                                                test=test,
                                                                                                ntl_type=ntl_type)

    patch_shape = X_train.shape[2:]
    t = X_train.shape[1]

    n_ex_train = X_train.shape[0]
    n_ex_test = X_test.shape[0]
    n_ex_val = X_val.shape[0]
    n_ex_ex = X_ex.shape[0]

    X_train = X_train.reshape(t * n_ex_train, *patch_shape)
    X_val = X_val.reshape(t * n_ex_val, *patch_shape)
    X_test = X_test.reshape(t * n_ex_test, *patch_shape)
    X_ex = X_ex.reshape(t * n_ex_ex, *patch_shape)

    y_train = y_train.reshape(t * n_ex_train)
    y_val = y_val.reshape(t * n_ex_val)
    y_test = y_test.reshape(t * n_ex_test)
    y_ex = y_ex.reshape(t * n_ex_ex)

    input_shape = X_train.shape[1:]
    output_shape = (1,)
    model = make_transformer_spat(input_shape, out_shape=output_shape,
                                   num_heads=num_heads, key_dim=key_dim, depth=depth)

    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                         decay_steps=(7000 // batch_size) * 2,
                                                                         decay_rate=1., staircase=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate_decay),
                  metrics=['mae'])

    callback_list, logs = callbacks('spat', test_fold=test)

    h = model.fit(X_train,
                  y_train,
                  epochs=epochs,
                  callbacks=callback_list,
                  batch_size=batch_size,
                  validation_data=(X_val, y_val))

    model.load_weights(logs + '/model-best')

    pd.DataFrame(h.history).to_csv(logs + '/history.csv')

    predictions = model.predict(X_test, verbose=1)
    predictions = predictions.reshape(n_ex_test, t)
    np.save(logs + '/predictions.npy', predictions)

    predictions = model.predict(X_ex)
    predictions = predictions.reshape(n_ex_ex, t)
    np.save(logs + '/ex_zone_pred.npy', predictions)

    del X_train, y_train, X_test, y_test, model


def run_effnet(train, val, test, ntl_type, epochs, batch_size, learning_rate, fine_tune_all, channels):

    (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_ex, y_ex) = load_train_val_test_ex(train=train, val=val,
                                                                                                test=test,
                                                                                                ntl_type=ntl_type)

    patch_shape = X_train.shape[2:]
    t = X_train.shape[1]

    n_ex_train = X_train.shape[0]
    n_ex_test = X_test.shape[0]
    n_ex_val = X_val.shape[0]
    n_ex_ex = X_ex.shape[0]

    X_train = X_train.reshape(t * n_ex_train, *patch_shape)
    X_val = X_val.reshape(t * n_ex_val, *patch_shape)
    X_test = X_test.reshape(t * n_ex_test, *patch_shape)
    X_ex = X_ex.reshape(t * n_ex_ex, *patch_shape)

    y_train = y_train.reshape(t * n_ex_train)
    y_val = y_val.reshape(t * n_ex_val)
    y_test = y_test.reshape(t * n_ex_test)
    y_ex = y_ex.reshape(t * n_ex_ex)

    if channels == 'rgb':
        X_train = X_train[..., [2, 1, 0]]
        X_val = X_val[..., [2, 1, 0]]
        X_test = X_test[..., [2, 1, 0]]
        X_ex = X_ex[..., [2, 1, 0]]

    input_shape = X_train.shape[1:]
    output_shape = (1,)
    model = efficient_net_pretrain(input_shape, out_shape=output_shape, fine_tune_all=fine_tune_all, channels=channels)

    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                         decay_steps=(7000 // batch_size) * 2,
                                                                         decay_rate=1., staircase=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate_decay),
                  metrics=['mae'])

    callback_list, logs = callbacks('effnet', test_fold=test)

    h = model.fit(X_train,
                  y_train,
                  epochs=epochs,
                  callbacks=callback_list,
                  batch_size=batch_size,
                  validation_data=(X_val, y_val))

    model.load_weights(logs + '/model-best')

    pd.DataFrame(h.history).to_csv(logs + '/history.csv')

    predictions = model.predict(X_test, verbose=1)
    predictions = predictions.reshape(n_ex_test, t)
    np.save(logs + '/predictions.npy', predictions)

    predictions = model.predict(X_ex)
    predictions = predictions.reshape(n_ex_ex, t)
    np.save(logs + '/ex_zone_pred.npy', predictions)

    del X_train, y_train, X_test, y_test, model


def run_baseline_cnn(train, val, test, ntl_type, epochs, batch_size, learning_rate, channels):

    (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_ex, y_ex) = load_train_val_test_ex(train=train, val=val,
                                                                                                test=test,
                                                                                                ntl_type=ntl_type)

    patch_shape = X_train.shape[2:]
    t = X_train.shape[1]

    n_ex_train = X_train.shape[0]
    n_ex_test = X_test.shape[0]
    n_ex_val = X_val.shape[0]
    n_ex_ex = X_ex.shape[0]

    X_train = X_train.reshape(t * n_ex_train, *patch_shape)
    X_val = X_val.reshape(t * n_ex_val, *patch_shape)
    X_test = X_test.reshape(t * n_ex_test, *patch_shape)
    X_ex = X_ex.reshape(t * n_ex_ex, *patch_shape)

    y_train = y_train.reshape(t * n_ex_train)
    y_val = y_val.reshape(t * n_ex_val)
    y_test = y_test.reshape(t * n_ex_test)
    y_ex = y_ex.reshape(t * n_ex_ex)

    if channels == 'rgb':
        X_train = X_train[..., [2, 1, 0]]
        X_val = X_val[..., [2, 1, 0]]
        X_test = X_test[..., [2, 1, 0]]
        X_ex = X_ex[..., [2, 1, 0]]

    input_shape = X_train.shape[1:]
    output_shape = (1,)
    model = baseline_cnn(input_shape, out_shape=output_shape)

    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                         decay_steps=(7000 // batch_size) * 2,
                                                                         decay_rate=1., staircase=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate_decay),
                  metrics=['mae'])

    callback_list, logs = callbacks('res_cnn', test_fold=test)

    h = model.fit(X_train,
                  y_train,
                  epochs=epochs,
                  callbacks=callback_list,
                  batch_size=batch_size,
                  validation_data=(X_val, y_val))

    model.load_weights(logs + '/model-best')

    pd.DataFrame(h.history).to_csv(logs + '/history.csv')

    predictions = model.predict(X_test, verbose=1)
    predictions = predictions.reshape(n_ex_test, t)
    np.save(logs + '/predictions.npy', predictions)

    predictions = model.predict(X_ex)
    predictions = predictions.reshape(n_ex_ex, t)
    np.save(logs + '/ex_zone_pred.npy', predictions)

    del X_train, y_train, X_test, y_test, model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-run_func")

    args = parser.parse_args()

    train = [[0, 1, 2, 3],
             [0, 1, 2, 4],
             [0, 1, 3, 4],
             [0, 2, 3, 4],
             [1, 2, 3, 4]]

    test = [4, 3, 2, 1, 0]
    val = [5, 5, 5, 5, 5]

    for tr, te, va in zip(train, test, val):

        if args.run_func == 'baseline':
            run_baseline(train=tr, val=va, test=te,
                         epochs=200, ntl_type='viirs', batch_size=64, learning_rate=5e-4,
                         num_heads=4, key_dim=72, depth=2)

        elif args.run_func == 'spat':
            run_spat(train=tr, val=va, test=te,
                         epochs=200, ntl_type='viirs', batch_size=64, learning_rate=5e-4,
                         num_heads=4, key_dim=72, depth=2)

        elif args.run_func == 'temp_only':
            run_temp_only(train=tr, val=va, test=te,
                          epochs=200, ntl_type='viirs', batch_size=64, learning_rate=5e-4,
                          num_heads=4, key_dim=72, depth=2)

        elif args.run_func == 'effnet_fine_tune_all':
            run_effnet(train=tr, val=va, test=te,
                       epochs=50, ntl_type='viirs', batch_size=64, learning_rate=5e-4,
                       fine_tune_all=True, channels='rgb')

        elif args.run_func == 'effnet_fine_tune_last':
            run_effnet(train=tr, val=va, test=te,
                       epochs=50, ntl_type='viirs', batch_size=64, learning_rate=5e-4,
                       fine_tune_all=False, channels='rgb')

        elif args.run_func == 'effnet_fine_tune_last_allc':
            run_effnet(train=tr, val=va, test=te,
                       epochs=50, ntl_type='viirs', batch_size=64, learning_rate=5e-4,
                       fine_tune_all=False, channels='all')

        elif args.run_func == 'effnet_fine_tune_all_allc':
            run_effnet(train=tr, val=va, test=te,
                       epochs=50, ntl_type='viirs', batch_size=64, learning_rate=5e-4,
                       fine_tune_all=True, channels='all')

        elif args.run_func == 'cnn':
            run_baseline_cnn(train=tr, val=va, test=te,
                             epochs=50, ntl_type='viirs', batch_size=64, learning_rate=5e-4,
                             channels='all')






