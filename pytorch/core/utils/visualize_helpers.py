import os
import cv2
import numpy as np

def tens2image(im):
    if im.size()[0] == 1:
        tmp = np.squeeze(im.numpy(), axis=0)
    else:
        tmp = im.numpy()
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))


def crop2fullmask(crop_mask, bbox, im=None, im_size=None, zero_pad=False, relax=0, mask_relax=True,
                  interpolation=cv2.INTER_CUBIC, scikit=False):
    if scikit:
        from skimage.transform import resize as sk_resize
    assert(not(im is None and im_size is None)), 'You have to provide an image or the image size'
    if im is None:
        im_si = im_size
    else:
        im_si = im.shape
    # Borers of image
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    # Bounding box of initial mask
    bbox_init = (bbox[0] + relax,
                 bbox[1] + relax,
                 bbox[2] - relax,
                 bbox[3] - relax)

    if zero_pad:
        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])
    else:
        assert((bbox == bbox_valid).all())
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    if scikit:
        crop_mask = sk_resize(crop_mask, (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), order=0, mode='constant').astype(crop_mask.dtype)
    else:
        crop_mask = cv2.resize(crop_mask, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)
    result_ = np.zeros(im_si)
    result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1] = \
        crop_mask[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1]

    result = np.zeros(im_si)
    if mask_relax:
        result[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1] = \
            result_[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1]
    else:
        result = result_

    return result


def overlay_mask(im, ma, colors=None, alpha=0.5):
    assert np.max(im) <= 1.0
    if colors is None:
        colors = np.load(os.path.join(os.path.dirname(__file__), '../dataset/', 'pascal_map.npy'))/255.
    else:
        colors = np.append([[0.,0.,0.]], colors, axis=0);

    if ma.ndim == 3:
        assert len(colors) >= ma.shape[0], 'Not enough colors'
    ma = ma.astype(np.bool)
    im = im.astype(np.float32)

    if ma.ndim == 2:
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[1, :3]   # np.array([0,0,255])/255.0
    else:
        fg = []
        for n in range(ma.ndim):
            fg.append(im * alpha + np.ones(im.shape) * (1 - alpha) * colors[1+n, :3])
    # Whiten background
    bg = im.copy()
    if ma.ndim == 2:
        bg[ma == 0] = im[ma == 0]
        bg[ma == 1] = fg[ma == 1]
        total_ma = ma
    else:
        total_ma = np.zeros([ma.shape[1], ma.shape[2]])
        for n in range(ma.shape[0]):
            tmp_ma = ma[n, :, :]
            total_ma = np.logical_or(tmp_ma, total_ma)
            tmp_fg = fg[n]
            bg[tmp_ma == 1] = tmp_fg[tmp_ma == 1]
        bg[total_ma == 0] = im[total_ma == 0]

    # [-2:] is s trick to be compatible both with opencv 2 and 3
    contours = cv2.findContours(total_ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(bg, contours[0], -1, (0.0, 0.0, 0.0), 1)

    return bg

def overlay_masks(im, masks, alpha=0.5):
    colors = np.load(os.path.join(os.path.dirname(__file__), '../utils/', 'pascal_map.npy'))/255.

    if isinstance(masks, np.ndarray):
        masks = [masks]

    assert len(colors) >= len(masks), 'Not enough colors'

    ov = im.copy()
    im = im.astype(np.float32)
    total_ma = np.zeros([im.shape[0], im.shape[1]])
    i = 1
    for ma in masks:
        ma = ma.astype(np.bool)
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[i, :3]   # np.array([0,0,255])/255.0
        i = i + 1
        ov[ma == 1] = fg[ma == 1]
        total_ma += ma

        # [-2:] is s trick to be compatible both with opencv 2 and 3
        contours = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(ov, contours[0], -1, (0.0, 0.0, 0.0), 1)
    ov[total_ma == 0] = im[total_ma == 0]

    return ov
