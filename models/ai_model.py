# app/models/ai_model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, concatenate, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import os

import boto3
import os

def download_model_from_s3(bucket_name, object_key, local_path):
    s3 = boto3.client('s3')
    if not os.path.exists(local_path):
        s3.download_file(bucket_name, object_key, local_path)
        
# First, recreate the loss functions and metrics used in the original model
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# Recreate the exact model architecture
def create_resnet50_unet_improved(input_shape=(512, 512, 1), n_classes=1):
    """
    Create an improved U-Net model with ResNet50 as encoder for mammography segmentation.
    """
    # Convert single channel to three channels by replication
    inputs = Input(shape=input_shape)
    x = Lambda(lambda x: tf.repeat(x, repeats=3, axis=-1))(inputs)

    # ResNet50 as encoder, without top layers
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=x)

    # Get layers from ResNet50 for skip connections
    skip1 = base_model.get_layer('conv1_relu').output         # 256x256
    skip2 = base_model.get_layer('conv2_block3_out').output   # 128x128
    skip3 = base_model.get_layer('conv3_block4_out').output   # 64x64
    skip4 = base_model.get_layer('conv4_block6_out').output   # 32x32

    # Bottleneck
    bottleneck = base_model.get_layer('conv5_block3_out').output  # 16x16

    # Decoder pathway with attention mechanism
    def attention_block(x, skip_connection, filters):
        g1 = Conv2D(filters, 1, padding='same')(x)
        g1 = BatchNormalization()(g1)

        x1 = Conv2D(filters, 1, padding='same')(skip_connection)
        x1 = BatchNormalization()(x1)

        g1_x1 = Activation('relu')(g1 + x1)
        psi = Conv2D(1, 1, padding='same')(g1_x1)
        psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)

        return psi * skip_connection

    # Decoder with attention
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    attention1 = attention_block(up1, skip4, 512)
    up1 = concatenate([up1, attention1])
    up1 = Conv2D(512, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(512, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Dropout(0.3)(up1)

    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up1)
    attention2 = attention_block(up2, skip3, 256)
    up2 = concatenate([up2, attention2])
    up2 = Conv2D(256, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(256, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Dropout(0.3)(up2)

    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up2)
    attention3 = attention_block(up3, skip2, 128)
    up3 = concatenate([up3, attention3])
    up3 = Conv2D(128, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(128, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Dropout(0.2)(up3)

    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up3)
    attention4 = attention_block(up4, skip1, 64)
    up4 = concatenate([up4, attention4])
    up4 = Conv2D(64, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(64, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Dropout(0.2)(up4)

    # Final upsampling to original image size
    up5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up4)
    up5 = Conv2D(32, (3, 3), padding='same')(up5)
    up5 = BatchNormalization()(up5)
    up5 = Activation('relu')(up5)
    up5 = Conv2D(32, (3, 3), padding='same')(up5)
    up5 = BatchNormalization()(up5)
    up5 = Activation('relu')(up5)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up5)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model with improved loss function
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=bce_dice_loss,
        metrics=['accuracy', dice_coef, tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
    )

    return model



# # Load the model
# def load_model():
#     model = create_resnet50_unet_improved(input_shape=(512, 512, 1))
#     model.load_weights("E:/Segmentation-BIRADS/BackEnd/routers/unet_mammography_best_model.h5")
#     return model

def load_model():
    # Set your S3 bucket and key
    S3_BUCKET = "aimodel-seg "
    S3_KEY = "https://aimodel-seg.s3.eu-north-1.amazonaws.com/unet_mammography_best_model.h5"
    LOCAL_MODEL_PATH = "routers/unet_mammography_best_model.h5"

    # Download the model from S3 if not present
    download_model_from_s3(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)

    # Build and load the model as before
    model = create_resnet50_unet_improved(input_shape=(256, 256, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(LOCAL_MODEL_PATH)
    return model

# Create a singleton instance
model = load_model()