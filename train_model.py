import os
import tensorflow as tf
import tensorflow_hub as hub
from keras import mixed_precision, layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from helper_functions import preprocess_img, plot_loss_curves

(train_data, test_data), ds_info = tfds.load(
    name="food101",
    split=["train", "validation"],
    shuffle_files=True,
    as_supervised=True,  # (data, label)
    with_info=True,
)

class_names = ds_info.features["label"].names

train_data = train_data.map(
    map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE
)
train_data = train_data.shuffle(buffer_size=1000).batch(64).prefetch(tf.data.AUTOTUNE)

test_data = test_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(64).prefetch(tf.data.AUTOTUNE)

mixed_precision.set_global_policy("mixed_float16")

checkpoint_path = "model_checkpoint/cp.ckpt"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=0,
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)

# Model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB5(include_top=False)
base_model.trainable = False


def create_model():
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs, training=False)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(len(class_names))(x)
    outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def fit_model(
    model,
    epochs=50,
    learning_rate=0.001,
    fine_tune=False,
    callbacks=early_stopping,
    history=None,
):
    initial_epochs = epochs

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_data,
        epochs=initial_epochs,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=len(test_data),
        callbacks=[callbacks],
    )

    if fine_tune:
        print("\nFine Tuning...\n")

        fine_tune_epochs = len(history.epoch) + epochs

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

        history = model.fit(
            train_data,
            epochs=fine_tune_epochs,
            steps_per_epoch=len(train_data),
            validation_data=test_data,
            validation_steps=len(test_data),
            initial_epoch=history.epoch[-1],
            callbacks=[callbacks],
        )

    return history


def layer_unfreezing(layers=100):
    for layer in base_model.layers[-layers:]:
        layer.trainable = True


if __name__ == "__main__":
    # plot_loss_curves(history_fine_tune_2)

    model = create_model()
    history = fit_model(model=model)
    layer_unfreezing()
    second_history = fit_model(
        model=model, learning_rate=0.0001, fine_tune=True, history=history
    )
    layer_unfreezing(0)
    third_history = fit_model(
        model=model, learning_rate=0.00001, fine_tune=True, history=second_history
    )

    model.save("EfficientNetB5.h5")
    model.save_weights("EfficientNeB5_weights")

    # model = tf.keras.models.load_model(
    #     "/home/dempavlo/vscodeprojects/tensorflow/models/EfficientNetB3_V2.h5"
    # )

    results = model.evaluate(test_data)
    print(results)
