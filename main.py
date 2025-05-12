# Read All dataset images
from glob import glob
image_path = 'images/'
mask_path = 'masks/'
input_img_paths = [i for i in glob(image_path + '*.png')]
target_img_paths = [i for i in glob(mask_path + '*.png')]

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_img_paths, target_img_paths, test_size=0.2, random_state=42)
