import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation
from keras.models import Model, save_model, load_model
from keras.optimizers import Adam
from keras import backend as K


RAND_SEED = 42
BATCH_SIZE = 16
IMAGE_SIZE = (768, 768)
IMAGE_SIZE_TRAINING = (IMAGE_SIZE[0] // 4, IMAGE_SIZE[1] // 4)
SMOOTH = 1e-5
np.random.seed(RAND_SEED)
tf.random.set_seed(RAND_SEED)
tf.keras.utils.set_random_seed(RAND_SEED)
# tf.config.experimental.enable_op_determinism()
DATA_DIR = 'data/'
WORK_DIR = 'data/'
IMAGE_DATA_DIR = os.path.join(DATA_DIR, 'train_v2')
MASKS_CACHE_DIR = os.path.join(WORK_DIR, 'masks_cache')

K.clear_session()
# from keras.mixed_precision import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
tf.keras.backend.set_floatx('float16')


def create_directories(WORK_DIR, MASKS_CACHE_DIR):
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
        print(f"Directory '{WORK_DIR}' created")
    else:
        print(f"Directory '{WORK_DIR}' already exists")

    if not os.path.exists(MASKS_CACHE_DIR):
        os.makedirs(MASKS_CACHE_DIR)
        print(f"Directory '{MASKS_CACHE_DIR}' created")
    else:
        print(f"Directory '{MASKS_CACHE_DIR}' already exists")


def load_clean_dataframe(DATA_DIR):
    df_truth = pd.read_csv(os.path.join(DATA_DIR, 'train_ship_segmentations_v2.csv'))
    # Group and cleanup data
    df_truth.dropna(subset=['EncodedPixels'], inplace=True)
    df_grouped = df_truth.groupby('ImageId')['EncodedPixels'].apply(lambda x: ' '.join(x)).reset_index()
    df_grouped['HasEncodedPixels'] = df_grouped['EncodedPixels'].apply(lambda x: not pd.isna(x) and x.strip() != "")

    return df_grouped


def split_dataframe(df_grouped, SUBSET_SIZE=0.03, fracs=[0.6, 0.2, 0.2]):
    
    # Split the dataset into subset and remaining dataset
    sss_initial = StratifiedShuffleSplit(n_splits=1, test_size=1-SUBSET_SIZE, random_state=RAND_SEED)
    for subset_index, _ in sss_initial.split(df_grouped, df_grouped['HasEncodedPixels']):
        df_subset = df_grouped.iloc[subset_index]

    # Pre-fill column for mask file names
    df_subset['MaskId'] = df_subset['ImageId'].apply(lambda x: (x.split('.')[0] + '.png'))

    # Now we have a stratified subset, let's split it into train/val/test
    fracs = {'train': 0.6, 'val': 0.2, 'test': 0.2}
    train_size = fracs['train'] / (fracs['train'] + fracs['val'])
    val_size = fracs['val'] / (fracs['val'] + fracs['test'])

    # Split the subset dataset into training and remaining dataset (for val/test)
    sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=1-train_size, random_state=RAND_SEED)
    for train_index, temp_index in sss_train_val.split(df_subset, df_subset['HasEncodedPixels']):
        df_train = df_subset.iloc[train_index]
        df_temp = df_subset.iloc[temp_index]

    # Use StratifiedShuffleSplit again for splitting the df_temp into validation and test sets
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_size, random_state=RAND_SEED)
    for val_index, test_index in sss_val_test.split(df_temp, df_temp['HasEncodedPixels']):
        df_val = df_temp.iloc[val_index]
        df_test = df_temp.iloc[test_index]

    # Now we have df_train, df_val, df_test with preserved class distribution
    print(f'Total samples in subset: {len(df_subset)} ({SUBSET_SIZE*100:.1f}% of the original dataset)')
    print(f'Training samples: {len(df_train)}')
    print(f'Validation samples: {len(df_val)}')
    print(f'Test samples: {len(df_test)}')
    
    return df_subset, df_train, df_val, df_test


"""
Pixels are encoded in pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc. 

Images with no ships have a blank value in the `EncodedPixels` column.
"""
# Get mask by decoding "run-length" EncodedPixels data
def get_mask_rle_decode(encoded_pixels, shape):
    s = encoded_pixels.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert one-indexed to zero-indexed
    ends = starts + lengths
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
        
    return mask.reshape(shape).T  # Need to transpose to align with the image shape


def precompute_masks_cache(df_subset, IMAGE_SIZE):
    for index, row in df_subset.iterrows():
        mask_file_path = os.path.join(MASKS_CACHE_DIR, row['MaskId'])
            
        # Check if the file already exists.
        if not os.path.exists(mask_file_path):
            # Check if the image has EncodedPixels/ships
            if row['HasEncodedPixels']:
                # Decode mask
                mask = get_mask_rle_decode(row['EncodedPixels'], shape=IMAGE_SIZE)
            # If there's no EncodedPixels/ships, create an array of zeroes.
            else:
                mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            
            # Convert to an image and save
            mask_image = Image.fromarray(mask.astype('uint8'))
            mask_image.save(mask_file_path)



def create_generator(df, target_size=IMAGE_SIZE_TRAINING):
    gen_args = dict(y_col=None,
                    class_mode=None,
                    batch_size=BATCH_SIZE,
                    target_size=target_size,
                    seed=RAND_SEED)
    datagen = ImageDataGenerator(rescale=1/255.)
    
    # Create Image Data Generators for images and masks
    image_gen = datagen.flow_from_dataframe(
        dataframe=df,
        directory=IMAGE_DATA_DIR,
        x_col='ImageId',
        **gen_args)
    
    mask_gen = datagen.flow_from_dataframe(
        dataframe=df,
        directory=MASKS_CACHE_DIR,
        x_col='MaskId',
        color_mode='grayscale',
        **gen_args)
    
    return zip(image_gen, mask_gen)


# Dice loss function
def dice_loss(y_true, y_pred, smooth=SMOOTH):
    
    return 1 - dice_coef(y_true, y_pred, smooth)


# Dice coef function for metrics
def dice_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]),'float32')
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]),'float32')
    intersection = tf.reduce_sum(tf.math.multiply(y_true_f,y_pred_f))

    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='HeNormal')(input_tensor)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='HeNormal')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    return x


def decoder_block(input_tensor, skip_tensor, num_filters):
    x = UpSampling2D((2, 2))(input_tensor)
    x = concatenate([x, skip_tensor])
    x = conv_block(x, num_filters)
    
    return x


def build_unet(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    e1 = conv_block(inputs, 8)
    p1 = MaxPooling2D((2, 2))(e1)
    e2 = conv_block(p1, 16)
    p2 = MaxPooling2D((2, 2))(e2)
    e3 = conv_block(p2, 32)
    p3 = MaxPooling2D((2, 2))(e3)

    # Bridge
    bridge = conv_block(p3, 64)

    # Decoder
    d3 = decoder_block(bridge, e3, 32)
    d2 = decoder_block(d3, e2, 16)
    d1 = decoder_block(d2, e1, 8)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d1)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


def main():

    create_directories(WORK_DIR, MASKS_CACHE_DIR)

    df_grouped = load_clean_dataframe(DATA_DIR)
    df_subset, df_train, df_val, df_test = split_dataframe(df_grouped, SUBSET_SIZE=0.03, fracs=[0.6, 0.2, 0.2])

    # df_subset = df_train + df_val + df_test
    precompute_masks_cache(df_subset, IMAGE_SIZE)

    train_gen = create_generator(df_train, IMAGE_SIZE_TRAINING)
    val_gen = create_generator(df_val, IMAGE_SIZE_TRAINING)
    test_gen = create_generator(df_test, IMAGE_SIZE_TRAINING)

    model = build_unet(input_shape=(*IMAGE_SIZE_TRAINING, 3))  # RGB
    model.compile(optimizer=Adam(), loss=dice_loss, metrics=[dice_coef])
    
    history = model.fit(train_gen, validation_data=val_gen, epochs=2, max_queue_size=16, workers=4, use_multiprocessing=True)

    # Save the history to a file
    with open(os.path.join(WORK_DIR, 'history.pkl'), 'wb') as file:
        pickle.dump(history.history, file)

    # Save the model
    save_model(model, os.path.join(WORK_DIR, 'model.h5'))


if __name__ == "__main__":
    main()
