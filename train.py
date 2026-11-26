import os
import time
import glob
import cv2
import numpy as np
import natsort
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from skimage.morphology import skeletonize
from PIL import Image

# ---------------------------
# USER CONFIG
# ---------------------------
BASE_DIR = r"C:\Users\nandh\Downloads\Resized_FIVES_Dataset 224"  # <-- set this
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_MASK_DIR  = os.path.join(BASE_DIR, "train", "masks")
TEST_IMAGE_DIR  = os.path.join(BASE_DIR, "test", "images")

SAVED_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "fin_attention_resunet_best_improved.keras")
RESULTS_DIR = os.path.join(BASE_DIR, "fin_test_predictions_improved")
os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_HEIGHT = 224
IMG_WIDTH  = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

BATCH_SIZE = 8
EPOCHS = 150
PATIENCE = 15     # Early stopping patience
SEED = 42

# ---------------------------
# PREPROCESSING
# ---------------------------
def preprocess_image_cv(img_bgr, clahe_clip=2.0, tile_grid=(8,8), gamma=1.15):
    """
    Input: img_bgr (uint8), BGR as read by cv2
    Returns: float32 image in [0,1], 3-channel (stacked) after CLAHE (green),
             gamma correction, z-score normalization and min-max to [0,1]
    """
    if img_bgr is None:
        raise ValueError("preprocess_image_cv received None image")
    
    img = img_bgr.copy()
    green = img[:, :, 1]  # Green channel

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile_grid)
    green = clahe.apply(green)

    # Gamma correction
    green_f = green.astype(np.float32) / 255.0
    green_f = np.power(green_f, 1.0 / gamma)
    green = (green_f * 255).astype(np.uint8)

    # Z-score normalization
    mean = np.mean(green)
    std = np.std(green) if np.std(green) > 0 else 1.0
    z = (green.astype(np.float32) - mean) / std

    # Min-max to [0,1]
    z = cv2.normalize(z, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Stack to 3 channels
    proc = np.stack([z, z, z], axis=-1).astype(np.float32)
    return proc

# ---------------------------
# DATA LOADING
# ---------------------------
def load_dataset(image_dir, mask_dir, img_h=IMG_HEIGHT, img_w=IMG_WIDTH):
    image_paths = natsort.natsorted(glob.glob(os.path.join(image_dir, ".")))
    mask_paths  = natsort.natsorted(glob.glob(os.path.join(mask_dir, ".")))

    images = []
    masks  = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print("Warning: couldn't read image:", p); continue
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
        proc = preprocess_image_cv(img)
        images.append(proc)

    for p in mask_paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            print("Warning: couldn't read mask:", p); continue
        m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        m_bin = (m > 127).astype(np.float32) # Use float32 for masks
        masks.append(np.expand_dims(m_bin, axis=-1))

    images = np.array(images, dtype=np.float32)
    masks  = np.array(masks, dtype=np.float32) # Ensure array is float32
    return images, masks

# ---------------------------
# AUGMENTATION (tf.data)
# ---------------------------
def tf_augment(img, msk):
    # Ensure types are float32
    img = tf.cast(img, tf.float32)
    msk = tf.cast(msk, tf.float32)

    # Combine image and mask to apply the same spatial transformations
    combined = tf.concat([img, msk], axis=-1)

    # Apply graph-safe random flips
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)

    # Split them back apart
    img = combined[..., :3]
    msk = combined[..., 3:]

    # Apply pixel-based augmentations to the image only
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.clip_by_value(img, 0.0, 1.0)

    return img, msk
    
# ---------------------------
# LOSS & METRICS
# ---------------------------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# ---------------------------
# MODEL (Attention ResU-Net)
# ---------------------------
def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    conv = layers.SeparableConv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.SeparableConv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.SeparableConv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)
    return res_path

def gating_signal(input, out_size, batch_norm=False):
    x = layers.SeparableConv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    theta_x = layers.SeparableConv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g = layers.SeparableConv2D(inter_shape, (1, 1), padding='same')(gating)
    concat_xg = layers.add([phi_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.SeparableConv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    
    upsample_psi = layers.UpSampling2D(
        size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2])
    )(sigmoid_xg)
    
    y = layers.multiply([upsample_psi, x])
    
    result = layers.SeparableConv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def Attention_ResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.5, batch_norm=True):
    FILTER_NUM = 16
    FILTER_SIZE = 3
    UP_SAMP_SIZE = 2
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Encoder
    conv_1 = res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_1 = layers.MaxPooling2D(pool_size=(2,2))(conv_1)
    conv_2 = res_conv_block(pool_1, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_2 = layers.MaxPooling2D(pool_size=(2,2))(conv_2)
    conv_3 = res_conv_block(pool_2, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_3 = layers.MaxPooling2D(pool_size=(2,2))(conv_3)
    conv_4 = res_conv_block(pool_3, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_4 = layers.MaxPooling2D(pool_size=(2,2))(conv_4)

    # Bridge (Atrous Spatial Pyramid Pooling inspired)
    f1 = layers.SeparableConv2D(128, (1, 1), activation='relu', padding='same')(pool_4)
    f2 = layers.SeparableConv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool_4)
    f3 = layers.SeparableConv2D(128, (3, 3), dilation_rate=4, activation='relu', padding='same')(pool_4)
    f4 = layers.SeparableConv2D(128, (3, 3), dilation_rate=8, activation='relu', padding='same')(pool_4)
    bridge = layers.concatenate([f1, f2, f3, f4], axis=3)

    # Decoder
    gating_1 = gating_signal(bridge, 8*FILTER_NUM, batch_norm)
    att_1 = attention_block(conv_4, gating_1, 8*FILTER_NUM)
    up_1 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(bridge)
    up_1 = layers.concatenate([up_1, att_1], axis=3)
    up_conv_1 = res_conv_block(up_1, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    gating_2 = gating_signal(up_conv_1, 4*FILTER_NUM, batch_norm)
    att_2 = attention_block(conv_3, gating_2, 4*FILTER_NUM)
    up_2 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(up_conv_1)
    up_2 = layers.concatenate([up_2, att_2], axis=3)
    up_conv_2 = res_conv_block(up_2, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)

    gating_3 = gating_signal(up_conv_2, 2*FILTER_NUM, batch_norm)
    att_3 = attention_block(conv_2, gating_3, 2*FILTER_NUM)
    up_3 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(up_conv_2)
    up_3 = layers.concatenate([up_3, att_3], axis=3)
    up_conv_3 = res_conv_block(up_3, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    gating_4 = gating_signal(up_conv_3, FILTER_NUM, batch_norm)
    att_4 = attention_block(conv_1, gating_4, FILTER_NUM)
    up_4 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(up_conv_3)
    up_4 = layers.concatenate([up_4, att_4], axis=3)
    up_conv_4 = res_conv_block(up_4, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    conv_final = layers.SeparableConv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_4)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)

    return Model(inputs, conv_final, name="AttentionResUNet")

# ---------------------------
# POST-PROCESSING
# ---------------------------
def refine_vessels(pred_binary_uint8, closing_kernel_size=5):
    """
    Input: pred_binary_uint8 (0/255)
    Returns: closed (0/255), skeleton (0/255)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    closed = cv2.morphologyEx(pred_binary_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    skeleton = skeletonize(closed > 0).astype(np.uint8) * 255
    return closed, skeleton

# ---------------------------
# MAIN SCRIPT
# ---------------------------
print("Loading dataset and preprocessing...")
X, Y = load_dataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
print(f"Loaded: X={X.shape}, Y={Y.shape}")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=SEED)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Build model
model = Attention_ResUNet(INPUT_SHAPE, NUM_CLASSES=1, dropout_rate=0.5, batch_norm=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=bce_dice_loss,
              metrics=[dice_coef, iou_coef, 'binary_accuracy'])

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ModelCheckpoint(SAVED_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

# Create tf.data datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (train_ds.shuffle(1024, seed=SEED)
                    .map(tf_augment, num_parallel_calls=AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .prefetch(AUTOTUNE))

val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Train the model
print("\nStarting training...")
start_time = time.time()
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    verbose=1)
print(f"Training finished. Time taken: {time.time() - start_time:.2f} seconds")

# Evaluate on the validation set
print("\nEvaluating on validation set...")
eval_res = model.evaluate(val_ds, verbose=1)
print(f"Validation Results - Loss: {eval_res[0]:.4f} | Dice: {eval_res[1]:.4f} | IoU: {eval_res[2]:.4f} | Accuracy: {eval_res[3]:.4f}")

# Inference on test set
print("\nLoading best model and running inference on test set...")
best_model = tf.keras.models.load_model(SAVED_MODEL_PATH, compile=False)

if not os.path.isdir(TEST_IMAGE_DIR):
    raise FileNotFoundError("TEST_IMAGE_DIR not found: " + TEST_IMAGE_DIR)

test_paths = natsort.natsorted(glob.glob(os.path.join(TEST_IMAGE_DIR, ".")))
print(f"Found {len(test_paths)} test images.")

THRESH = 0.5
closing_kernel = 5

for p in test_paths:
    base = os.path.splitext(os.path.basename(p))[0]
    img_bgr = cv2.imread(p)
    if img_bgr is None:
        print(f"Skipping unreadable file: {p}"); continue
        
    img_resized = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))
    img_proc = preprocess_image_cv(img_resized)
    img_input = np.expand_dims(img_proc, axis=0)

    # Use the best loaded model for prediction
    pred_raw = best_model.predict(img_input, verbose=0)[0, :, :, 0]
    
    # Post-processing and saving
    pred_raw_255 = (pred_raw * 255).astype(np.uint8)
    pred_binary = (pred_raw > THRESH).astype(np.uint8) * 255
    closed, skeleton = refine_vessels(pred_binary, closing_kernel_size=closing_kernel)

    Image.fromarray(pred_raw_255).save(os.path.join(RESULTS_DIR, f"{base}_raw.png"), dpi=(600,600))
    Image.fromarray(pred_binary).save(os.path.join(RESULTS_DIR, f"{base}_binary.png"), dpi=(600,600))
    Image.fromarray(closed).save(os.path.join(RESULTS_DIR, f"{base}_closed.png"), dpi=(600,600))
    Image.fromarray(skeleton).save(os.path.join(RESULTS_DIR, f"{base}_skeleton.png"), dpi=(600,600))
    print(f"Saved predictions for: {base}")

print(f"\nAll done. Results saved in: {RESULTS_DIR}")

print(f"\nAll done. Results saved in: {RESULTS_DIR}")