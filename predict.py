import tensorflow as tf

import numpy as np
from skimage.color import gray2rgb
from model import get_model
from scipy.ndimage import binary_fill_holes


def img_to_tiles(img, tgt_shape):
    '''Takes an RGB or RGBA ndarray image (img) and returns a list of
    overlappingtiles of shape=tgt_shape'''
    tgt_h, tgt_w, tgt_ch = tgt_shape
    img_h, img_w = img.shape[:2]

    if len(img.shape) == 2:  # control for grayscale inputs
        img = gray2rgb(img)
    img = img[..., :tgt_ch]  # remove alpha channel if present

    nh_tiles = int(np.ceil(img_h / tgt_h))
    nw_tiles = int(np.ceil(img_w / tgt_w))

    if (nh_tiles*nw_tiles) == 1:
        tiles = [img]
    else:
        tiles = []
        for h in range(nh_tiles):
            if nh_tiles == 1:
                h_i = 0
                h_f = img_h
            else:
                h_i = int(h/(nh_tiles-1) * (img_h-tgt_h))
                h_f = int(h_i + tgt_h)
            for w in range(nw_tiles):
                if nw_tiles == 1:
                    w_i = 0
                    w_f = img_w
                else:
                    w_i = int(w/(nw_tiles-1)*(img_w-tgt_w))
                    w_f = int(w_i+tgt_w)
                tiles.append(img[h_i:h_f, w_i:w_f])
    return tiles


def tiles_to_img(tiles, img_shape):
    '''Takes a list of overlapping tiles and composes them into a single image
    with shape img_shape'''
    img_h, img_w = img_shape[:2]
    tgt_h, tgt_w = tiles[0].shape

    nh_tiles = int(np.ceil(img_h / tgt_h))
    nw_tiles = int(np.ceil(img_w / tgt_w))

    if (nh_tiles*nw_tiles) == 1:
        img = tiles[0]
    else:
        img = np.zeros(img_shape, dtype=np.uint8)
        counter = 0
        for h in range(nh_tiles):
            if nh_tiles == 1:
                h_i = 0
                h_f = img_h
            else:
                h_i = int(h/(nh_tiles-1) * (img_h-tgt_h))
                h_f = int(h_i + tgt_h)
            for w in range(nw_tiles):
                if nw_tiles == 1:
                    w_i = 0
                    w_f = img_w
                else:
                    w_i = int(w/(nw_tiles-1)*(img_w-tgt_w))
                    w_f = int(w_i+tgt_w)
                img[h_i:h_f, w_i:w_f] = tiles[counter]
                counter += 1
    return img


def predict(imgs, IDs=None, **predict_args):
    '''Predicts nuclei masks from a list of images'''

    if IDs is None:
        IDs = [i for i in range(len(imgs))]

    # Break imgs into a list of 128x128x3 tiles
    img_tile_list = []
    ID_tile_list = []
    for i in range(len(imgs)):
        img = imgs[i]
        img_tiles = img_to_tiles(img, (128, 128, 3))
        img_tile_list.extend(img_tiles)

        ID_tile_list.extend([IDs[i]]*len(img_tiles))  # keep track of IDs

    ID_tile_array = np.array(ID_tile_list)  # to array

    # Predict nuclei masks
    model = get_model()
    X = tf.convert_to_tensor(np.array(img_tile_list))
    preds = model.predict(X, **predict_args)
    preds_bin = (preds > 0.5).astype(np.uint8)

    # Put tiles back together
    preds_list = []
    for ID in IDs:
        # Isolate tiles from each ID
        tiles = preds_bin[ID_tile_array == ID][..., 0]

        # Get original image shape
        img_ind = np.where(np.array(IDs) == ID)[0][0]
        shape = imgs[img_ind].shape[:2]

        # Back to original shape
        pred_img = tiles_to_img(tiles, shape)

        # Basic cleaning:
        pred_img_filled = binary_fill_holes(pred_img)

        # Append to list
        preds_list.append(pred_img_filled)

    return preds_list
