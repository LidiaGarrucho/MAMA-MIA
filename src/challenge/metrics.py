import numpy as np
import SimpleITK as sitk
from typing import Tuple, Union, List
from scipy.spatial import cKDTree

def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask

def _check_all_same(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if not len(i) == len(input_list[0]):
                return False
            all_same = all(i[j] == input_list[0][j] for j in range(len(i)))
            if not all_same:
                return False
        return True

def read_segmentation_masks(image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
    """
    Reads one or more segmentation masks and extracts image arrays and metadata.

    Returns:
        - stacked_images (np.ndarray): Stacked segmentation arrays in nnU-Net-compatible shape.
        - info (dict): Metadata with spacing, origin, and direction.
    """
    images, spacings, origins, directions, nnunet_spacings = [], [], [], [], []

    for f in image_fnames:
        img = sitk.ReadImage(f)
        arr = sitk.GetArrayFromImage(img)

        spacings.append(img.GetSpacing())
        origins.append(img.GetOrigin())
        directions.append(img.GetDirection())

        if arr.ndim == 2:
            arr = arr[None, None]
            max_spacing = max(spacings[-1])
            nnunet_spacings.append([max_spacing * 999] + list(spacings[-1])[::-1])
        elif arr.ndim == 3:
            arr = arr[None]
            nnunet_spacings.append(list(spacings[-1])[::-1])
        elif arr.ndim == 4:
            nnunet_spacings.append(list(spacings[-1])[::-1][1:])
        else:
            raise RuntimeError(f"Unexpected number of dimensions ({arr.ndim}) in file: {f}")

        images.append(arr)
        nnunet_spacings[-1] = list(np.abs(nnunet_spacings[-1]))

    # Validation checks
    def check_consistency(values, name, error=True):
        if not _check_all_same(values):
            print(f"{'ERROR' if error else 'WARNING'}! Not all input images have the same {name}!")
            print(f"{name.capitalize()}s:\n{values}")
            print(f"Image files:\n{image_fnames}")
            if error:
                raise RuntimeError()

    check_consistency([img.shape for img in images], "shape")
    check_consistency(spacings, "spacing")
    check_consistency(origins, "origin", error=False)
    check_consistency(directions, "direction", error=False)
    check_consistency(nnunet_spacings, "spacing_for_nnunet")

    stacked_images = np.vstack(images).astype(np.float32)
    return stacked_images, {
        "sitk_stuff": {
            "spacing": spacings[0],
            "origin": origins[0],
            "direction": directions[0],
        },
        "spacing": nnunet_spacings[0],
    }

def hausdorff_distance(image0, image1, method='95perc'):
    """Compute Hausdorff distance between segmentation masks."""
    if isinstance(image0, str):
        image0 = sitk.ReadImage(image0, sitk.sitkUInt8)
    else:
        image0 = sitk.GetImageFromArray(image0.astype(np.uint8))
    if isinstance(image1, str):
        image1 = sitk.ReadImage(image1, sitk.sitkUInt8)
    else:
        image1 = sitk.GetImageFromArray(image1.astype(np.uint8))
    image0_array = sitk.GetArrayFromImage(sitk.LabelContour(image0))
    image1_array = sitk.GetArrayFromImage(sitk.LabelContour(image1))

    a_points = np.argwhere(image0_array > 0)
    b_points = np.argwhere(image1_array > 0)

    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )
    if method == 'standard':
        return max(max(fwd), max(bwd))
    elif method == 'modified':
        return max(np.mean(fwd), np.mean(bwd))
    elif method == '95perc':
        return max(np.percentile(fwd, 95), np.percentile(bwd, 95))

def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn

def compute_segmentation_metrics(reference_file, prediction_file,
                    label=1, ignore_label=None, hd_max=150) -> dict:
    """Compute Dice, IoU, and Hausdorff metrics for segmentation."""
    if isinstance(reference_file, str):
        seg_ref, _ = read_segmentation_masks((reference_file,))
    else:
        seg_ref = reference_file
    if isinstance(prediction_file, str):
        seg_pred, _ = read_segmentation_masks((prediction_file,))
    else:
        seg_pred = prediction_file

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    metrics = {}
    mask_ref = region_or_label_to_mask(seg_ref, label)
    mask_pred = region_or_label_to_mask(seg_pred, label)
    tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
    dice = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0
    hausdorff_dist = hausdorff_distance(reference_file, prediction_file, method='95perc')
    norm_hausdorff = hausdorff_dist / hd_max if hausdorff_dist != np.inf else 1
    metrics['DSC'] = dice
    metrics['NormHD'] = norm_hausdorff
        
    return metrics
