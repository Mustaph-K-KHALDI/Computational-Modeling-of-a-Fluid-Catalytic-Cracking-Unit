from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Flatten, RepeatVector, Reshape,Conv1D, MaxPool1D, concatenate, Input, Dropout
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
import tensorflow as tf
from utils import creat_path
import subprocess


def get_gpu_utilization(device_index=0):
    """
    Call nvidia-smi to get the current GPU utilization percentage.
    Returns a float (e.g. 45.0 for 45%).
    """
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            f'--id={device_index}',
            '--query-gpu=utilization.gpu',
            '--format=csv,noheader,nounits'
        ], stderr=subprocess.STDOUT)
        return float(out.decode().strip())
    except subprocess.CalledProcessError as e:
        # if nvidia-smi isn’t available or fails, return NaN
        return float('nan')

class GpuMemoryLogger(tf.keras.callbacks.Callback):
    def __init__(self, device_index=0):
        super().__init__()
        self.device_index = device_index
        # will hold dicts of {'epoch', 'current_mb', 'peak_mb', 'util_pct'}
        self.mem_stats = []

    def on_epoch_end(self, epoch, logs=None):
        # 1) Memory info from TF
        info = tf.config.experimental.get_memory_info(f'GPU:{self.device_index}')
        current_mb = info['current'] / (1024 ** 2)
        peak_mb    = info['peak']    / (1024 ** 2)

        # 2) Utilization via nvidia-smi
        util_pct = get_gpu_utilization(self.device_index)

        # 3) Record it
        stats = {
            'epoch':        epoch,
            'current_mb':   current_mb,
            'peak_mb':      peak_mb,
            'util_pct':     util_pct,
        }
        self.mem_stats.append(stats)

        # 4) Inject into logs so it shows up in history.history
        if logs is None:
            logs = {}
        logs['gpu_current_mb'] = current_mb
        logs['gpu_peak_mb']    = peak_mb
        logs['gpu_util_pct']   = util_pct


def dense_model(window, units=10, activation="elu"):
    """
    Simple dense model felxible in input/output size.
    """

    model = Sequential(
        [
            # Shape: (time, features) => (time*features)
            Flatten(),
            Dense(
                units=units,
                activation=activation,
                # activity_regularizer=regularizers.l2(1e-3),
            ),
            Dense(
                units=(window.label_width * window.number_label_features)
            ),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            Reshape([window.label_width, window.number_label_features]),
        ]
    )
    return model


def conv_flex_model(window, filters=10, kernel_size=1, activation="elu"):
    """
    Flexible conv model in input/output size.
    """
    model = Sequential(
        [
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=activation,
                # activity_regularizer=regularizers.l2(1e-3),
            ),
            MaxPool1D(pool_size=2),
            Flatten(),
            Dense(
                units=(window.label_width * window.number_label_features)
            ),
            Reshape(
                [window.label_width, window.number_label_features]),
        ]
    )
    return model


def LSTM_model(l2r1,l1r1,units,lr,dropout,dropout_rate, window):
    """
    Simplest LSTM model with an LSTM layer that returns only the last cell output.
    forecasts any input/output size flexibly
    """
    r1 = l1_l2(l1=l1r1,l2=l2r1)
    model = Sequential()
    model.add(LSTM(
        units=units,
        return_sequences=False,
        kernel_regularizer=r1,
        bias_regularizer=r1,
        recurrent_regularizer=r1))
    if dropout:
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(
        units=(window.label_width * window.number_label_features)))
    model.add(Reshape(
                [window.label_width, window.number_label_features]))

    learning_rate = lr
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.metrics.RootMeanSquaredError()])

    return model

def Stacked_LSTM_model(l2r1,l1r1,units,lr,dropout,stack,window):

    r1 = l1_l2(l1=l1r1,l2=l2r1)
    model = Sequential()
    
    # Tune the number of layers.
    for i in range(stack):
        model.add(
            LSTM(
                # Tune number of units separately.
                units=units,
                return_sequences=True,
                kernel_regularizer=r1,
                bias_regularizer=r1,
                recurrent_regularizer=r1
            )
        )
        if dropout:
            model.add(Dropout(rate=0.20))
    model.add(LSTM(
        units=units,
        return_sequences=False,
        kernel_regularizer=r1,
        bias_regularizer=r1,
        recurrent_regularizer=r1))
    if dropout:
            model.add(Dropout(rate=0.20))
    model.add(Dense(
        units=(window.label_width * window.number_label_features)))
    model.add(Reshape(
                [window.label_width, window.number_label_features]))

    learning_rate = lr
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.metrics.RootMeanSquaredError()])

    return model

def LSTM_model_encoder_decoder(window, encoder_units=10, decoder_units=10):
    """
    LSTM model with 2 LSTM layers: a many to one and a one to many layers.
    """
    model = Sequential(
        [
            LSTM(encoder_units, return_sequences=False),
            Dropout(0.1),
            RepeatVector(window.label_width),
            LSTM(decoder_units, return_sequences=True),
            Dropout(0.1),
            Dense(window.number_label_features)])
    return model

def CNN_LSTM(l2r1,l1r1,l2r2,l1r2,units,filters,kz,pz,lr,dropout,droupout_p, window):

    r1 = l1_l2(l1=l1r1,l2=l2r1)
    r2 = l1_l2(l1=l1r2,l2=l2r2)
    input_layer = Input(shape=(window.input_width, 18))
    conv = Conv1D(
            filters=filters,
            kernel_size=kz,
            bias_regularizer=r1,
            kernel_regularizer=r1)(input_layer)
    conv = Conv1D(
            filters=filters,
            kernel_size=kz,
            bias_regularizer=r1,
            kernel_regularizer=r1)(conv)
    conv = MaxPool1D(pool_size=pz)(conv)
    lstm = LSTM(
                # Tune number of units separately.
                units=units,
                return_sequences=True,
                kernel_regularizer=r2,
                bias_regularizer=r2,
                recurrent_regularizer=r2,
            )(conv)
    if dropout:
        lstm = Dropout(rate=droupout_p)(lstm)
    lstm = LSTM(
                # Tune number of units separately.
                units=units,
                return_sequences=False,
                kernel_regularizer=r2,
                bias_regularizer=r2,
                recurrent_regularizer=r2,
            )(lstm)
    if dropout:
        lstm = Dropout(rate=droupout_p)(lstm)

    dense = Dense(units=(window.label_width *
                  window.number_label_features))(lstm)
    output_layer = Reshape(
        [window.label_width, window.number_label_features])(dense)
    model = Model([input_layer], [output_layer])
    learning_rate = lr
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.metrics.RootMeanSquaredError()])
    return model

def BiLSTM_model(l2r1,l1r1,units,lr,dropout,droupout_p,stack, window):

    r1 = l1_l2(l1=l1r1,l2=l2r1)
    model = Sequential()
    #model.add(Flatten())
    # Tune the number of layers.
    for i in range(stack):
        model.add(
            Bidirectional(LSTM(
                # Tune number of units separately.
                units=units,
                return_sequences=True,
                kernel_regularizer=r1,
                bias_regularizer=r1
            ))
        )
        if dropout:
            model.add(Dropout(rate=droupout_p))
    model.add(Bidirectional(LSTM(
        units=units,
        return_sequences=False,
        kernel_regularizer=r1,
        bias_regularizer=r1)))
    if dropout:
            model.add(Dropout(rate=droupout_p))
    model.add(Dense(
        units=(window.label_width * window.number_label_features)))
    model.add(Reshape([window.label_width, window.number_label_features]))

    learning_rate = lr
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.metrics.RootMeanSquaredError()])

    return model

def MHNN(window,n_reac,n_reg,n_frac):
    """
    Flexible conv model in input/output size.
    """
    reac_input = Input(shape=(window.input_width, n_reac), name="reac")
    reg_input = Input(shape=(window.input_width, n_reg), name="reg")
    frac_input = Input(shape=(window.input_width, n_frac), name="frac")

    reac_features_n = LSTM(15, return_sequences=True)(reac_input)

    reg_features_n = LSTM(13, return_sequences=True)(reg_input)

    frac_features_n = LSTM(10,)(frac_input)

    fcc_input = concatenate([reac_features_n, reg_features_n])

    fcc = LSTM(10)(fcc_input)

    fccu_input = concatenate([fcc, frac_features_n])

    dense = Dense(units=(window.label_width *
                  window.number_label_features))(fccu_input)
    output_layer = Reshape(
        [window.label_width, window.number_label_features], name='fccu')(dense)
    model = Model(inputs=[reac_input, reg_input,
                  frac_input], outputs=output_layer)

    return model

def compile_and_fit(
        model,
        X=None,
        y=None,
        val_X=None,
        val_y=None,
        patience_es=50,
        patience_r=20,
        batch_size=128,
        max_epochs=100,
        v=1,
        lr=1e-3,
        model_name='model'):

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience_es,
        mode="min",
        min_delta=0.0001,
        restore_best_weights=True,
        verbose=v)
    path = creat_path(model_name)
    mc = ModelCheckpoint(
        path+"/"+model_name+'.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=v)

    rlrop = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=patience_r)
    
    gpu_cb  = GpuMemoryLogger(device_index=0)

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        metrics=[tf.metrics.RootMeanSquaredError()],
    )
    history = model.fit(
        X,
        y,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(val_X, val_y),
        callbacks=[es, mc, rlrop, gpu_cb],
        verbose=v
    )
    
    history.history['gpu_peak_mb']    = [m['peak_mb']    for m in gpu_cb.mem_stats]
    history.history['gpu_current_mb'] = [m['current_mb'] for m in gpu_cb.mem_stats]
    history.history['gpu_util_pct']   = [m['util_pct']   for m in gpu_cb.mem_stats]

    return history
