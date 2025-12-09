# Image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# Data loading function
def load_data(image_dir, mask_dir, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    images = []
    masks = []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(os.path.join(image_dir, img_file))
        img = cv2.resize(img, (img_width, img_height))
        img = img / 255.0  # Normalize
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_width, img_height))
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        mask = mask / 255.0  # Normalize
        images.append(img)
        masks.append(mask)
    return np.array(images), np.array(masks)

# Paths of augmented dataset
image_dir = '/kaggle/input/udait-aug-images'
mask_dir = '/kaggle/input/udait-aug-GT'
images, masks = load_data(image_dir, mask_dir)

# Train/Test split (manual test data path as per the requirement)
train_data = images
train_masks = masks

# Manually specify the test data and test masks paths
test_image_dir = '/kaggle/input/udait/test-imges'
test_mask_dir = '/kaggle/input/udait/test-gt'

test_images, test_masks = load_data(test_image_dir, test_mask_dir)

