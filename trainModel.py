import os, argparse, glob, datetime
import numpy as np
import tensorflow as tf
import pandas as pd  


AUTOTUNE = tf.data.AUTOTUNE

FER_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def fer_from_class_folders_root(data_dir, seed=42, val_split=0.1, test_split=0.1):
    """
    Ожидает структуру:
      data_dir/<class_name>/*.jpg|png|...
    где class_name: 0..6 или angry/disgust/.../neutral

    Делит на train/val/test по долям.
    """
    def class_to_index(name: str) -> int:
        name_l = name.lower()
        if name_l.isdigit():
            idx = int(name_l)
            if 0 <= idx <= 6:
                return idx
        if name_l in FER_LABELS:
            return FER_LABELS.index(name_l)
        raise ValueError(f"Неизвестный класс папки: {name}")

    paths, labels = [], []
    for cls in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        idx = class_to_index(cls)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.gif"):
            for p in glob.glob(os.path.join(cls_dir, ext)):
                paths.append(p)
                labels.append(idx)

    if len(paths) == 0:
        raise FileNotFoundError("В корне data_dir не найдено изображений в папках-классах.")

    paths = np.array(paths)
    labels = np.array(labels, dtype=np.int32)


    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(paths))
    paths, labels = paths[perm], labels[perm]
    n = len(paths)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Слишком большие val_split/test_split для размера датасета.")

    tr_p, tr_y = paths[:n_train], labels[:n_train]
    va_p, va_y = paths[n_train:n_train + n_val], labels[n_train:n_train + n_val]
    te_p, te_y = paths[n_train + n_val:], labels[n_train + n_val:]

    return (tr_p, tr_y), (va_p, va_y), (te_p, te_y)


def augment_image(img, training: bool):
    img = tf.cast(img, tf.float32) / 255.0

    if not training:
        return img


    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.10)
    img = tf.image.random_contrast(img, 0.85, 1.15)

    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)

    img = tf.image.rot90(img, k)
    img = tf.image.resize_with_crop_or_pad(img, 54, 54)
    img = tf.image.random_crop(img, size=[48, 48, 1])


    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.03)
    img = img + noise
    img = tf.clip_by_value(img, 0.0, 1.0)

    return img


def build_dataset_from_paths(paths, y, batch_size, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(min(len(paths), 20000), reshuffle_each_iteration=True)

    def decode(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=1, expand_animations=False)
        img = tf.image.resize(img, [48, 48], method="bilinear")
        img = tf.cast(img, tf.float32)
        img.set_shape([48, 48, 1])
        return img

    def map_fn(path, label):
        img = decode(path)
        img = augment_image(img, training)
        label = tf.one_hot(label, depth=7)
        return img, label

    ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def residual_block(x, filters, weight_decay=1e-4):
    shortcut = x

    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, padding="same", use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)
    return x


def make_model(weight_decay=1e-4):
    """
    Улучшенная CNN для FER:
    - Residual-блоки
    - L2-регуляризация
    - GlobalAveragePooling
    """
    inputs = tf.keras.Input(shape=(48, 48, 1))
    x = inputs

    x = tf.keras.layers.Conv2D(
        64, 3, padding="same", use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = residual_block(x, 64, weight_decay)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = residual_block(x, 128, weight_decay)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    x = residual_block(x, 256, weight_decay)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Conv2D(
        256, 3, padding="same", use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(7, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def compute_class_weights(labels, num_classes=7):
    """
    class_weight для борьбы с дисбалансом.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    class_weights = {i: float(weights[i]) for i in range(num_classes)}
    print("Class counts:", counts)
    print("Class weights:", class_weights)
    return class_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Папка, где сразу лежат подпапки-классы (angry/happy/... или 0/1/...). "
             "Если не задано — берётся ./trainFER рядом со скриптом."
    )
    parser.add_argument("--out", type=str, default=None,
                        help="Куда сохранить .h5 лучшую модель (если не задано — рядом со скриптом)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    enable_gpu_memory_growth()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "trainFER")
    data_dir = args.data_dir if args.data_dir else default_data_dir

    out_path = args.out if args.out else os.path.join(script_dir, "emotion_model_fer2013_best.h5")

    run_name = "FER_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(script_dir, "runs", run_name)
    os.makedirs(logdir, exist_ok=True)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"data_dir не папка: {data_dir}\n"
            f"Положи датасет (подпапки-эмоции) в: {default_data_dir}\n"
            f"или запусти с параметром: --data_dir \"...\""
        )

    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = fer_from_class_folders_root(
        data_dir, seed=args.seed, val_split=args.val_split, test_split=args.test_split
    )
    train_ds = build_dataset_from_paths(tr_p, tr_y, args.batch, training=True)
    val_ds   = build_dataset_from_paths(va_p, va_y, args.batch, training=False)
    test_ds  = build_dataset_from_paths(te_p, te_y, args.batch, training=False) if len(te_p) else None

    print("Loaded root class-folders dataset:", data_dir)
    print("Counts:", len(tr_p), len(va_p), len(te_p))

    class_weights = compute_class_weights(tr_y, num_classes=7)

    model = make_model(weight_decay=1e-4)

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2"),
        ]
    )

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        out_path, monitor="val_acc", mode="max",
        save_best_only=True, save_weights_only=False
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_acc", mode="max",
        patience=20, restore_best_weights=True
    )
    reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
    )
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch",
        profile_batch=0
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt, early, reduce, tb],
        verbose=1,
        class_weight=class_weights
    )

    print("\nBest model saved to:", out_path)
    best = tf.keras.models.load_model(out_path)

    if test_ds is not None:
        res = best.evaluate(test_ds, verbose=1)
        print("Test:", dict(zip(best.metrics_names, res)))
    else:
        res = best.evaluate(val_ds, verbose=1)
        print("Val (as test):", dict(zip(best.metrics_names, res)))


if __name__ == "__main__":
    main()
