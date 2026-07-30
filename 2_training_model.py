import tensorflow as tf
from tensorflow.keras import layers, models

def build_3d_unet(input_shape=(64, 64, 64, 7)):
    inputs = layers.Input(input_shape)

    # 编码器（Contracting Path）
    conv1 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D((2, 2, 2))(conv1)

    conv2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling3D((2, 2, 2))(conv2)

    conv3 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling3D((2, 2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)

    # 解码器（Expansive Path）
    up5 = layers.UpSampling3D(size=(2, 2, 2))(conv4)
    merge5 = layers.concatenate([conv3, up5], axis=-1)
    conv5 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(merge5)
    conv5 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = layers.UpSampling3D(size=(2, 2, 2))(conv5)
    merge6 = layers.concatenate([conv2, up6], axis=-1)
    conv6 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(merge6)
    conv6 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.UpSampling3D(size=(2, 2, 2))(conv6)
    merge7 = layers.concatenate([conv1, up7], axis=-1)
    conv7 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge7)
    conv7 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv7)

    # 输出层
    output = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)

    model = models.Model(inputs, output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 创建模型
model = build_3d_unet()

# 输出模型概况
model.summary()
