# src/train.py
# -*- coding: utf-8 -*-
"""
Script để huấn luyện mô hình U-Net cho bài toán semantic segmentation
tách chủ thể người nhảy.
"""
import os
import tensorflow as tf
from matplotlib import pyplot as plt

# Import các hàm và hằng số từ các file khác trong project
from data_loader import get_datasets, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
from model import build_unet, NUM_CLASSES # NUM_CLASSES thường là 1 cho binary segmentation

# --- Cấu hình quá trình huấn luyện ---
# Bạn có thể điều chỉnh các giá trị này để thử nghiệm
LEARNING_RATE = 1e-2 # Tốc độ học ban đầu
BATCH_SIZE = 8       # Số lượng mẫu trong một batch (điều chỉnh tùy theo bộ nhớ GPU)
                     # Nếu bạn đã đặt batch_size trong get_datasets, giá trị này có thể không cần
                     # thiết khi gọi model.fit nếu dùng tf.data.Dataset đã batch.
                     # Tuy nhiên, nó vẫn cần để tính steps_per_epoch.
NUM_EPOCHS = 200      # Số lần duyệt qua toàn bộ tập huấn luyện
                     # Bắt đầu với số nhỏ (20-50) để xem xu hướng, sau đó có thể tăng lên.

# Đường dẫn lưu mô hình
MODEL_SAVE_PATH = 'saved_models/' # Lưu ở thư mục gốc của project, trong saved_models
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
BEST_MODEL_FILEPATH = os.path.join(MODEL_SAVE_PATH, 'unet_dancer_segmentation_best.keras') # Dùng .keras

# Đường dẫn lưu logs cho TensorBoard
TENSORBOARD_LOG_DIR = 'logs/training_logs' # Lưu ở thư mục gốc của project, trong logs
if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.makedirs(TENSORBOARD_LOG_DIR)

def plot_training_history(history, model_name="U-Net"):
    """
    Vẽ đồ thị loss và Mean IoU từ đối tượng history của Keras.
    """
    # Lấy các giá trị từ history
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    # Kiểm tra xem 'mean_iou' hay 'val_mean_iou' có trong history không
    # Tên metric có thể khác nhau tùy theo cách bạn đặt tên khi compile
    # Ví dụ: 'mean_io_u' (do Keras tự đổi tên) hoặc tên bạn đặt trong tf.keras.metrics.MeanIoU(name='my_iou')
    iou_metric_name = None
    val_iou_metric_name = None

    for key in history.history.keys():
        if 'iou' in key.lower() and 'val' not in key.lower():
            iou_metric_name = key
        if 'iou' in key.lower() and 'val' in key.lower():
            val_iou_metric_name = key
            
    if not iou_metric_name or not val_iou_metric_name:
        print("Cảnh báo: Không tìm thấy metric 'MeanIoU' trong history. Sẽ chỉ vẽ loss.")
        mean_iou = []
        val_mean_iou = []
    else:
        mean_iou = history.history.get(iou_metric_name, [])
        val_mean_iou = history.history.get(val_iou_metric_name, [])

    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(14, 5))

    # Đồ thị Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Đồ thị Mean IoU (nếu có)
    if mean_iou and val_mean_iou:
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, mean_iou, label=f'Training {iou_metric_name}')
        plt.plot(epochs_range, val_mean_iou, label=f'Validation {val_iou_metric_name}')
        plt.title(f'{model_name} - Training and Validation Mean IoU')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # Lưu đồ thị (tùy chọn)
    # plt.savefig(os.path.join('../results/', 'training_history.png'))
    plt.show()


def main():
    """
    Hàm chính để thực hiện toàn bộ quá trình huấn luyện.
    """
    print("--- Bắt đầu quá trình chuẩn bị dữ liệu ---")
    # Lấy datasets từ data_loader
    # Sử dụng BATCH_SIZE đã định nghĩa ở trên cho get_datasets
    train_dataset, val_dataset, num_train_samples, num_val_samples = get_datasets(
        batch_size=BATCH_SIZE,
        test_size=0.2 # Giữ 20% cho validation
    )

    if train_dataset is None or val_dataset is None:
        print("Lỗi: Không thể tạo datasets. Kết thúc chương trình.")
        return

    print(f"Số lượng mẫu huấn luyện: {num_train_samples}")
    print(f"Số lượng mẫu kiểm định: {num_val_samples}")
    print("--- Hoàn thành chuẩn bị dữ liệu ---")

    # --- Xây dựng mô hình ---
    print("\n--- Bắt đầu xây dựng mô hình U-Net ---")
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model = build_unet(input_shape=input_shape, num_classes=NUM_CLASSES)
    print("--- Hoàn thành xây dựng mô hình U-Net ---")
    model.summary() # In ra tóm tắt kiến trúc

    # --- Compile mô hình ---
    # Định nghĩa optimizer, loss, và metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_function = tf.keras.losses.BinaryCrossentropy() # Vì output là sigmoid (0-1)
    
    # MeanIoU là metric quan trọng cho segmentation
    # num_classes=2 vì chúng ta có 2 lớp: nền (0) và chủ thể (1)
    metrics = [
        'accuracy', # Độ chính xác pixel
        tf.keras.metrics.MeanIoU(num_classes=2, name='mean_iou') # Đặt tên để dễ truy cập trong history
    ]

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    print("\n--- Đã compile mô hình ---")

    # --- Định nghĩa Callbacks ---
    print("\n--- Định nghĩa Callbacks ---")
    # Lưu lại model tốt nhất dựa trên val_mean_iou
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_FILEPATH,
        save_weights_only=False, # Lưu toàn bộ mô hình
        monitor='val_mean_iou',  # Theo dõi val_mean_iou
        mode='max',              # Chế độ 'max' vì IoU càng cao càng tốt
        save_best_only=True,     # Chỉ lưu model tốt nhất
        verbose=1
    )

    # Dừng sớm nếu val_mean_iou không cải thiện
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_mean_iou',
        patience=10, # Số epochs chờ đợi nếu không có cải thiện đáng kể
        mode='max',
        verbose=1,
        restore_best_weights=True # Khôi phục trọng số từ epoch tốt nhất khi dừng
    )

    # Giảm learning rate nếu val_mean_iou không cải thiện
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mean_iou',
        factor=0.2, # Giảm LR đi 0.2 lần (LR_new = LR_old * factor)
        patience=5, # Số epochs chờ đợi
        mode='max',
        verbose=1,
        min_lr=1e-7 # Learning rate tối thiểu
    )

    # TensorBoard callback để theo dõi quá trình huấn luyện
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=TENSORBOARD_LOG_DIR,
        histogram_freq=1, # Ghi lại histogram của trọng số mỗi epoch
        write_graph=True,
        write_images=False # Có thể đặt True để xem ảnh đầu vào/đầu ra trong TensorBoard
    )

    callbacks_list = [
        checkpoint_callback,
        early_stopping_callback,
        reduce_lr_callback,
        tensorboard_callback
    ]
    print("--- Callbacks đã được định nghĩa ---")

    # --- Bắt đầu Huấn luyện ---
    print(f"\n--- Bắt đầu huấn luyện mô hình với {NUM_EPOCHS} epochs ---")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate ban đầu: {LEARNING_RATE}")

    # Tính steps_per_epoch và validation_steps
    # Cần thiết nếu dùng tf.data.Dataset hoặc generator
    steps_per_epoch = num_train_samples // BATCH_SIZE
    if num_train_samples % BATCH_SIZE != 0: # Nếu có phần dư
        steps_per_epoch += 1
    
    validation_steps = num_val_samples // BATCH_SIZE
    if num_val_samples % BATCH_SIZE != 0: # Nếu có phần dư
        validation_steps += 1

    history = model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1 # In ra thông tin huấn luyện sau mỗi epoch
    )

    print("\n--- Hoàn thành quá trình huấn luyện! ---")

    # --- (Tùy chọn) Lưu lại mô hình cuối cùng (không nhất thiết nếu đã có ModelCheckpoint) ---
    # final_model_path = os.path.join(MODEL_SAVE_PATH, 'unet_dancer_segmentation_final.keras')
    # model.save(final_model_path)
    # print(f"Đã lưu mô hình cuối cùng tại: {final_model_path}")

    # --- Vẽ đồ thị lịch sử huấn luyện ---
    print("\n--- Vẽ đồ thị lịch sử huấn luyện ---")
    plot_training_history(history, model_name="U-Net Dancer Segmentation")

if __name__ == '__main__':
    # Kiểm tra xem BASE_DATA_PATH trong data_loader có được cấu hình đúng không
    # Bằng cách import và thử truy cập nó (nếu nó là biến toàn cục)
    # Hoặc bạn có thể thêm một hàm kiểm tra cấu hình trong data_loader
    try:
        from data_loader import BASE_DATA_PATH as data_loader_base_path
        if 'PATH_TO_YOUR_DancingPeopleSegmentation_FOLDER' in data_loader_base_path:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! LỖI CẤU HÌNH: Bạn CHƯA THAY ĐỔI `BASE_DATA_PATH` trong file `src/data_loader.py`      !!!")
            print("!!! Vui lòng mở file src/data_loader.py và cập nhật đường dẫn này trước khi huấn luyện. !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            exit()
    except ImportError:
        print("Lỗi: Không thể import BASE_DATA_PATH từ data_loader.py. Kiểm tra file.")
        exit()
    except AttributeError:
         print("Cảnh báo: Không tìm thấy biến BASE_DATA_PATH trong data_loader.py để kiểm tra.")


    # Chạy hàm main để bắt đầu huấn luyện
    main()
