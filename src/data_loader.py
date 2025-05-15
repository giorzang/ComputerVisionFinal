# src/data_loader.py
# -*- coding: utf-8 -*-
"""
Script để tải, tiền xử lý và tạo pipeline dữ liệu cho mô hình segmentation.
"""
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf # Sử dụng tf.data cho pipeline hiệu quả

# --- Các hằng số và cấu hình ---
IMG_WIDTH = 512  # Chiều rộng ảnh đầu vào cho mô hình
IMG_HEIGHT = 512 # Chiều cao ảnh đầu vào cho mô hình
IMG_CHANNELS = 3 # Số kênh màu của ảnh gốc (ví dụ: RGB)

# !!! QUAN TRỌNG: Thay đổi đường dẫn này cho phù hợp với máy của bạn !!!
# Ví dụ: 'D:/BaiTapLon/DancingPeopleSegmentation/data/'
# Hoặc: '/home/user/projects/DancingPeopleSegmentation/data/'
BASE_DATA_PATH = 'D:/ZangDev/CodeByZang/Phenikaa/ComputerVision/Project/data/'

IMAGE_DIR = os.path.join(BASE_DATA_PATH, 'images')
MASK_DIR = os.path.join(BASE_DATA_PATH, 'masks')

def load_image_and_mask_paths(image_dir, mask_dir):
    """
    Lấy danh sách đường dẫn đầy đủ đến các file ảnh và mask tương ứng.
    Hàm này giả định tên file trong 'images' và 'masks' là giống nhau.
    """
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir)) # Giả sử tên file tương ứng

    # Kiểm tra sơ bộ xem số lượng file có khớp không
    if len(image_filenames) != len(mask_filenames):
        print(f"Cảnh báo: Số lượng file ảnh ({len(image_filenames)}) và mask ({len(mask_filenames)}) không khớp!")
        # Có thể thêm logic để chỉ lấy các cặp file khớp tên nếu cần

    # Tạo đường dẫn đầy đủ
    image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]
    mask_paths = [os.path.join(mask_dir, fname) for fname in mask_filenames]
    
    # Chỉ giữ lại các cặp file mà cả ảnh và mask đều tồn tại
    # và có tên cơ sở giống nhau (nếu cần kiểm tra kỹ hơn)
    valid_image_paths = []
    valid_mask_paths = []
    for img_p, mask_p in zip(image_paths, mask_paths):
        # Kiểm tra tên cơ sở có giống nhau không (ví dụ: image1.jpg và image1.jpg)
        # Hoặc bạn có thể có quy tắc đặt tên khác, ví dụ image1_mask.jpg
        if os.path.basename(img_p) == os.path.basename(mask_p):
             if os.path.exists(img_p) and os.path.exists(mask_p):
                valid_image_paths.append(img_p)
                valid_mask_paths.append(mask_p)
        # else: # Ví dụ nếu mask có hậu tố _mask
        #    base_img_name, _ = os.path.splitext(os.path.basename(img_p))
        #    expected_mask_name = f"{base_img_name}_mask.jpg" # Hoặc .png
        #    if os.path.basename(mask_p) == expected_mask_name:
        #         if os.path.exists(img_p) and os.path.exists(mask_p):
        #            valid_image_paths.append(img_p)
        #            valid_mask_paths.append(mask_p)


    if len(valid_image_paths) != len(valid_mask_paths) or not valid_image_paths:
         print(f"Sau khi kiểm tra, số lượng cặp ảnh/mask hợp lệ không khớp hoặc bằng 0.")
         print(f"Ảnh hợp lệ: {len(valid_image_paths)}, Mask hợp lệ: {len(valid_mask_paths)}")
         # Có thể cần dừng chương trình hoặc xử lý lỗi ở đây nếu không có dữ liệu
         # return [],[] # Trả về rỗng nếu không có cặp nào

    print(f"Tìm thấy {len(valid_image_paths)} cặp ảnh/mask hợp lệ.")
    return valid_image_paths, valid_mask_paths


def preprocess_image(image_path, target_height=IMG_HEIGHT, target_width=IMG_WIDTH):
    """
    Đọc, resize và chuẩn hóa ảnh gốc.
    """
    img = cv.imread(image_path, cv.IMREAD_COLOR) # Đọc ảnh màu
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh: {image_path}")
        return None
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # Chuyển sang RGB (Matplotlib/TF thường dùng RGB)
    img = cv.resize(img, (target_width, target_height), interpolation=cv.INTER_AREA)
    img = img / 255.0 # Chuẩn hóa giá trị pixel về [0, 1]
    return img.astype(np.float32)

def preprocess_mask(mask_path, target_height=IMG_HEIGHT, target_width=IMG_WIDTH):
    """
    Đọc, resize và chuẩn hóa ảnh mask.
    Mask đầu ra sẽ có giá trị 0 (nền) hoặc 1 (chủ thể).
    """
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE) # Đọc ảnh mask dưới dạng xám
    if mask is None:
        print(f"Lỗi: Không thể đọc mask: {mask_path}")
        return None
    mask = cv.resize(mask, (target_width, target_height), interpolation=cv.INTER_NEAREST)
    # Nhị phân hóa mask: các giá trị > 127 (hoặc ngưỡng phù hợp) thành 1, còn lại là 0
    mask = (mask > 127).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1) # Thêm chiều kênh: (H, W, 1)
    return mask

def tf_dataset_generator(image_paths, mask_paths, batch_size, shuffle=True):
    """
    Tạo một tf.data.Dataset để tải và tiền xử lý dữ liệu theo batch.
    """
    # Tạo dataset từ các đường dẫn file
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if shuffle:
        # Xáo trộn dữ liệu (quan trọng cho tập huấn luyện)
        # Lấy tổng số lượng mẫu để buffer_size hoạt động tốt
        buffer_size = len(image_paths)
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    def load_and_preprocess(img_path_tensor, mask_path_tensor):
        # Hàm này cần được bọc bởi tf.py_function để dùng cv2 bên trong tf.data
        def _py_load_and_preprocess(img_path, mask_path):
            # Chuyển tensor đường dẫn sang string Python
            img_p = img_path.numpy().decode('utf-8')
            mask_p = mask_path.numpy().decode('utf-8')
            
            img = preprocess_image(img_p)
            mask = preprocess_mask(mask_p)
            
            if img is None or mask is None:
                # Xử lý trường hợp không đọc được file (ví dụ trả về mảng rỗng hoặc raise error)
                # Ở đây ta sẽ tạo mảng rỗng để pipeline không bị dừng, nhưng cần log lỗi
                print(f"Cảnh báo: Lỗi khi tải {img_p} hoặc {mask_p}. Trả về mảng rỗng.")
                return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32), \
                       np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
            return img, mask

        # Sử dụng tf.py_function
        image, mask = tf.py_function(
            _py_load_and_preprocess,
            [img_path_tensor, mask_path_tensor],
            [tf.float32, tf.float32] # Kiểu dữ liệu trả về
        )
        # Đặt lại shape vì tf.py_function làm mất thông tin shape
        image.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        mask.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
        return image, mask

    # Áp dụng hàm load_and_preprocess cho từng cặp (ảnh, mask)
    # num_parallel_calls=tf.data.AUTOTUNE giúp tối ưu hóa việc tải song song
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Tạo batch
    dataset = dataset.batch(batch_size)

    # Prefetch để tối ưu hiệu năng (CPU chuẩn bị dữ liệu trong khi GPU huấn luyện)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

# --- Hàm chính để chuẩn bị dữ liệu ---
def get_datasets(test_size=0.2, batch_size=32, random_state=42):
    """
    Tải đường dẫn, chia train/val, và tạo tf.data.Dataset.
    """
    all_image_paths, all_mask_paths = load_image_and_mask_paths(IMAGE_DIR, MASK_DIR)

    if not all_image_paths or not all_mask_paths:
        print("Không có dữ liệu để xử lý. Kết thúc.")
        return None, None, 0, 0 # Trả về None nếu không có dữ liệu

    # Chia tập train và validation
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        all_image_paths, all_mask_paths,
        test_size=test_size,      # Tỷ lệ cho tập validation
        random_state=random_state # Để đảm bảo kết quả chia là nhất quán
    )

    print(f"Số lượng mẫu huấn luyện: {len(train_img_paths)}")
    print(f"Số lượng mẫu kiểm định: {len(val_img_paths)}")

    # Tạo tf.data.Dataset
    train_dataset = tf_dataset_generator(train_img_paths, train_mask_paths, batch_size, shuffle=True)
    val_dataset = tf_dataset_generator(val_img_paths, val_mask_paths, batch_size, shuffle=False) # Không cần shuffle tập val

    return train_dataset, val_dataset, len(train_img_paths), len(val_img_paths)


# --- Ví dụ cách sử dụng (có thể chạy riêng file này để kiểm tra) ---
if __name__ == '__main__':
    print("Đang kiểm tra data_loader...")
    
    # Thay đổi đường dẫn BASE_DATA_PATH ở đầu file này cho đúng!
    if 'PATH_TO_YOUR_DancingPeopleSegmentation_FOLDER' in BASE_DATA_PATH:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! LỖI CẤU HÌNH: Bạn CHƯA THAY ĐỔI `BASE_DATA_PATH` trong src/data_loader.py !!!")
        print("!!! Vui lòng mở file src/data_loader.py và cập nhật đường dẫn này.         !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

    BATCH_SIZE_TEST = 4 # Batch size nhỏ để test
    train_ds, val_ds, num_train_samples, num_val_samples = get_datasets(batch_size=BATCH_SIZE_TEST)

    if train_ds and val_ds:
        print(f"\nĐã tạo thành công train_dataset và val_dataset.")
        print(f"Số mẫu huấn luyện: {num_train_samples}, Số mẫu kiểm định: {num_val_samples}")

        print("\nKiểm tra một batch từ train_dataset:")
        for images, masks in train_ds.take(1): # Lấy 1 batch để kiểm tra
            print(f"  Shape của batch ảnh: {images.shape}") # (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            print(f"  Shape của batch mask: {masks.shape}")  # (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1)
            print(f"  Kiểu dữ liệu ảnh: {images.dtype}")
            print(f"  Kiểu dữ liệu mask: {masks.dtype}")
            print(f"  Giá trị min/max của ảnh trong batch: {np.min(images.numpy()):.2f} / {np.max(images.numpy()):.2f}")
            print(f"  Giá trị min/max của mask trong batch: {np.min(masks.numpy()):.0f} / {np.max(masks.numpy()):.0f}")
            print(f"  Các giá trị duy nhất trong một mask mẫu: {np.unique(masks.numpy()[0])}")


            # Hiển thị ảnh và mask đầu tiên trong batch
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(images.numpy()[0]) # Ảnh đã được normalize về [0,1] và là RGB
            plt.title("Ảnh mẫu từ batch")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(masks.numpy()[0][:, :, 0], cmap='gray') # Mask là (H, W, 1), lấy kênh 0
            plt.title("Mask mẫu từ batch")
            plt.axis('off')
            plt.show()
            break # Chỉ kiểm tra 1 batch
    else:
        print("Không thể tạo datasets. Vui lòng kiểm tra lỗi ở trên.")

