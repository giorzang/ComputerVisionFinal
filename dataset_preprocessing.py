import os
from PIL import Image

# Thư mục ảnh và mask
BASE_DATA_PATH = '/media/giorzang/Data-SSD/ZangDev/CodeByZang/Phenikaa/ComputerVision/Project/bigdata/'
image_dir = os.path.join(BASE_DATA_PATH, 'images')
mask_dir = os.path.join(BASE_DATA_PATH, 'masks')

# Duyệt qua tất cả các file ảnh .jpg trong thư mục images
for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    # Kiểm tra xem file mask tương ứng có tồn tại không
    if os.path.exists(mask_path):
        try:
            # Mở ảnh và mask
            with Image.open(image_path) as img, Image.open(mask_path) as msk:
                if img.size != msk.size or img.size != (540, 960):
                    print(f"Không khớp kích thước: {filename} -> xóa")

                    # Xóa cả ảnh và mask
                    os.remove(image_path)
                    os.remove(mask_path)
                # print(img.size, msk.size)

        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {e}")
    else:
        print(f"Không tìm thấy mask tương ứng với {filename}")

    
