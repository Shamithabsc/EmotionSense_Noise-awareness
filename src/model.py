from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_lstm(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    """Compact CNN-LSTM for fixed-length MFCC feature sequences."""
    inputs = keras.Input(shape=input_shape, name="mfcc_sequence")
    x = layers.Reshape((input_shape[0], input_shape[1]))(inputs)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(96, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, dropout=0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_probabilities")(x)
    model = keras.Model(inputs, outputs, name="EmotionSense_CNN_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_improved_cnn_bilstm(
    input_shape: tuple[int, int, int],
    num_classes: int,
    learning_rate: float = 4e-4,
    label_smoothing: float = 0.08,
) -> keras.Model:
    """Enhanced CNN-BiLSTM with Learnable Temporal Attention and Multi-Pooling.

    Improvements:
      - 2 Conv1D blocks with L2 regularization and SpatialDropout1D to suppress speaker memorization
      - Bidirectional LSTM (64 units) capturing both forward and reverse temporal dynamics
      - Learnable Temporal Attention pooling (Bahdanau-style frame salience weights)
      - Multi-pooling concatenation: Attention context + Global Average Pool + Global Max Pool (384-dim)
      - Dense classification head with BatchNorm, Dropout(0.35), and L2 regularization
      - Label-smoothed Categorical Crossentropy for handling perceptual ambiguity in emotion classes
    """
    inputs = keras.Input(shape=input_shape, name="features_sequence")
    x = layers.Reshape((input_shape[0], input_shape[1]))(inputs)

    # Block 1: Local spectral-temporal acoustic feature extraction
    x = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(0.12)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Block 2: Higher-level formant transition modeling
    x = layers.Conv1D(128, kernel_size=3, padding="same",
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(0.12)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Bidirectional LSTM with full sequence output
    lstm_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2,
                    kernel_regularizer=keras.regularizers.l2(1e-4))
    )(x)  # shape: (batch, 50, 128)

    # Learnable Temporal Attention Mechanism
    attn_dense = layers.Dense(64, activation="tanh")(lstm_out)
    attn_weights = layers.Dense(1, activation="softmax", name="temporal_attention_weights")(attn_dense)
    context = layers.Dot(axes=1)([attn_weights, lstm_out])
    context_flat = layers.Flatten()(context)  # (batch, 128)

    # Multi-pooling: Attention salient frames + Global average prosody + Global peak bursts
    avg_pool = layers.GlobalAveragePooling1D()(lstm_out)  # (batch, 128)
    max_pool = layers.GlobalMaxPooling1D()(lstm_out)      # (batch, 128)
    combined = layers.Concatenate()([context_flat, avg_pool, max_pool])  # (batch, 384)

    # Dense classification head
    d = layers.Dense(96, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(combined)
    d = layers.BatchNormalization()(d)
    d = layers.Dropout(0.35)(d)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_probabilities")(d)

    model = keras.Model(inputs, outputs, name="EmotionSense_Improved_BiLSTM_Attention")
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing) if label_smoothing > 0 else "categorical_crossentropy"
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model

