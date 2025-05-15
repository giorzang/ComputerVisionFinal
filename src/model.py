# src/model.py
# -*- coding: utf-8 -*-
"""
Script định nghĩa kiến trúc mô hình U-Net cho bài toán semantic segmentation.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

# --- Các hằng số (có thể import từ data_loader.py hoặc định nghĩa lại nếu cần) ---
# Giả sử chúng ta sẽ truyền input_shape vào hàm build_unet
# IMG_WIDTH = 128
# IMG_HEIGHT = 128
# IMG_CHANNELS = 3
NUM_CLASSES = 1 # Output là mask nhị phân (0 hoặc 1 cho chủ thể/nền)

def conv_block(input_tensor, num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same'):
    """
    Một khối convolutional cơ bản gồm 2 lớp Conv2D.
    """
    # Lớp Convolutional thứ nhất
    x = Conv2D(num_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(input_tensor)
    # Lớp Convolutional thứ hai
    x = Conv2D(num_filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(x)
    return x

def encoder_block(input_tensor, num_filters, pool_size=(2, 2), dropout_rate=0.1):
    """
    Một khối trong phần Encoder của U-Net.
    Bao gồm một conv_block, một lớp Dropout (tùy chọn), và một lớp MaxPooling.
    """
    # Khối convolutional
    c = conv_block(input_tensor, num_filters)
    # Dropout (giúp giảm overfitting)
    if dropout_rate > 0:
        c = Dropout(dropout_rate)(c)
    # Lớp MaxPooling để giảm kích thước không gian
    p = MaxPooling2D(pool_size)(c)
    return c, p # Trả về cả output của conv_block (cho skip connection) và output của pooling

def decoder_block(input_tensor, skip_features, num_filters, kernel_size=(3,3), strides=(2,2), padding='same', dropout_rate=0.1):
    """
    Một khối trong phần Decoder của U-Net.
    Bao gồm một lớp Conv2DTranspose (upsampling), nối với skip_features,
    và một conv_block.
    """
    # Upsampling bằng Conv2DTranspose
    up = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding=padding)(input_tensor)
    # Nối (concatenate) với đặc trưng từ skip connection
    # Đảm bảo skip_features có cùng kích thước không gian với 'up'
    # Nếu không, bạn có thể cần Cropping2D hoặc padding trên skip_features
    merged = concatenate([up, skip_features])
    # Khối convolutional
    c = conv_block(merged, num_filters)
    # Dropout
    if dropout_rate > 0:
        c = Dropout(dropout_rate)(c)
    return c

def build_unet(input_shape, num_classes=NUM_CLASSES):
    """
    Xây dựng kiến trúc mô hình U-Net.

    Args:
        input_shape (tuple): Shape của ảnh đầu vào, ví dụ (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).
        num_classes (int): Số lớp đầu ra. Cho binary segmentation (chủ thể/nền), num_classes=1.

    Returns:
        tensorflow.keras.models.Model: Mô hình U-Net đã được xây dựng.
    """
    inputs = Input(input_shape)

    # --- Encoder (Contracting Path / Đường xuống) ---
    # Block 1
    c1, p1 = encoder_block(inputs, num_filters=32, dropout_rate=0.1) # Kích thước ví dụ: 128 -> 64
    # Block 2
    c2, p2 = encoder_block(p1, num_filters=64, dropout_rate=0.1)   # 64 -> 32
    # Block 3
    c3, p3 = encoder_block(p2, num_filters=128, dropout_rate=0.2)   # 32 -> 16
    # Block 4
    c4, p4 = encoder_block(p3, num_filters=256, dropout_rate=0.2)   # 16 -> 8

    # --- Bottleneck (Phần cổ chai) ---
    # Không có pooling ở đây
    b = conv_block(p4, num_filters=512) # Kích thước ví dụ: 8x8x1024
    if 0.3 > 0: # Giữ dropout_rate nhất quán
        b = Dropout(0.3)(b)


    # --- Decoder (Expansive Path / Đường lên) ---
    # Block 1 (upsample từ bottleneck, nối với c4)
    d1 = decoder_block(b, skip_features=c4, num_filters=256, dropout_rate=0.2)   # 8 -> 16
    # Block 2 (upsample từ d1, nối với c3)
    d2 = decoder_block(d1, skip_features=c3, num_filters=128, dropout_rate=0.2)  # 16 -> 32
    # Block 3 (upsample từ d2, nối với c2)
    d3 = decoder_block(d2, skip_features=c2, num_filters=64, dropout_rate=0.1)  # 32 -> 64
    # Block 4 (upsample từ d3, nối với c1)
    d4 = decoder_block(d3, skip_features=c1, num_filters=32, dropout_rate=0.1)   # 64 -> 128

    # --- Output Layer ---
    # Sử dụng Conv2D với kernel_size (1,1) và hàm kích hoạt sigmoid cho binary segmentation.
    # Số filter bằng num_classes (trong trường hợp này là 1).
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(d4)

    # Tạo mô hình
    model = Model(inputs=[inputs], outputs=[outputs], name="UNet_DancerSegmentation")
    return model

# --- Ví dụ cách sử dụng (có thể chạy riêng file này để kiểm tra kiến trúc) ---
if __name__ == '__main__':
    # Định nghĩa input shape ví dụ (nên giống với cấu hình trong data_loader.py)
    # Bạn có thể import các hằng số này từ data_loader.py nếu muốn
    IMG_HEIGHT_TEST = 128
    IMG_WIDTH_TEST = 128
    IMG_CHANNELS_TEST = 3 # Ảnh màu RGB
    
    example_input_shape = (IMG_HEIGHT_TEST, IMG_WIDTH_TEST, IMG_CHANNELS_TEST)
    
    print(f"Đang xây dựng mô hình U-Net với input_shape: {example_input_shape}")
    
    # Xây dựng mô hình
    unet_model = build_unet(input_shape=example_input_shape)
    
    # In ra tóm tắt kiến trúc mô hình
    # Điều này rất hữu ích để kiểm tra số lượng tham số và shape của các lớp.
    unet_model.summary()
    
    # (Tùy chọn) Vẽ kiến trúc mô hình ra file ảnh (cần cài đặt pydot và graphviz)
    # try:
    #     tf.keras.utils.plot_model(unet_model, to_file='unet_model_architecture.png', show_shapes=True)
    #     print("\nĐã lưu kiến trúc mô hình vào file 'unet_model_architecture.png'")
    # except ImportError:
    #     print("\nKhông thể vẽ kiến trúc mô hình. Cần cài đặt pydot và graphviz.")
    #     print("Lệnh cài đặt: pip install pydot graphviz")

