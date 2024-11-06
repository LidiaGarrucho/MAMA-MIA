import os
import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage import restoration

def read_mri_phase_from_patient_id(images_folder, patient_id, phase=0):
    '''Read the MRI phase from the selected patient ID using SimpleITK:
    images_folder: str - the path to the folder containing the MRI images
    patient_id: str - the patient id
    phase: int - the phase of the MRI image (0: pre-contrast, 1: first post-contrast, 2: second post-contrast, ...)
    '''
    phase_sitk = sitk.ReadImage(f'{images_folder}/{patient_id}/{patient_id}_000{phase}.nii.gz', sitk.sitkFloat32)
    return phase_sitk

def read_segmentation_from_patient_id(segmentations_folder, patient_id):
    '''Read the segmentation from the selected patient ID using SimpleITK:
    images_folder: str - the path to the folder containing the primary tumor segmentations
    patient_id: str - the patient id
    '''
    mask_sitk = sitk.ReadImage(f'{segmentations_folder}/{patient_id}.nii.gz', sitk.sitkUInt8)
    return mask_sitk

def get_image_orientation_from_direction(img_sitk):
    """
    img_sitk = SimpleITK image
    return string orientation
    """
    filter = sitk.DICOMOrientImageFilter()
    filter.Execute(img_sitk)
    orientation = filter.GetOrientationFromDirectionCosines(img_sitk.GetDirection())
    return orientation

def bias_correction_sitk(image_sitk, otsu_threshold=False, shrink_factor=0):
    """Apply N4 Bias Correction."""
    if shrink_factor:
        # N4BiasFieldCorrectionImageFilter takes too long to run, shrink image
        mask_breast = sitk.OtsuThreshold(image_sitk, 0, 1)
        shrinked_image_sitk = sitk.Shrink(image_sitk, [shrink_factor] * image_sitk.GetDimension())
        shrinked_mask_breast = sitk.Shrink(mask_breast, [shrink_factor] * mask_breast.GetDimension())
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        tmp_image = corrector.Execute(shrinked_image_sitk, shrinked_mask_breast)
        log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)
        corrected_image_sitk = image_sitk / sitk.Exp(log_bias_field)
    else:
        initial_img = image_sitk
        # Cast to float to enable bias correction to be used
        tmp_image = sitk.Cast(image_sitk, sitk.sitkFloat64)
        # Set zeroes to a small number to prevent division by zero
        tmp_image = sitk.GetArrayFromImage(tmp_image)
        tmp_image[tmp_image == 0] = np.finfo(float).eps
        tmp_image = sitk.GetImageFromArray(tmp_image)
        tmp_image.CopyInformation(initial_img)
        if otsu_threshold:
            maskImage = sitk.OtsuThreshold(tmp_image, 0, 1)
        # Apply image bias correction using N4 bias correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        if otsu_threshold:
            corrected_image_sitk = corrector.Execute(tmp_image, maskImage)
        else:
            corrected_image_sitk = corrector.Execute(tmp_image)

    return corrected_image_sitk

def nlmeans_denoise_sitk(image_sitk, patch_size=5, patch_distance=6, h=0.8):
    """
    Denoises a DCE-MRI image using Non-Local Means (NLMeans) filtering.
    
    Parameters:
        image_sitk (SimpleITK.Image): Input DCE-MRI image.
        patch_size (int): Size of the patches used for denoising.
        patch_distance (int): Maximal distance in pixels where to search patches used for denoising.
        h (float): Cut-off distance (higher h means more smoothing).
        
    Returns:
        SimpleITK.Image: Denoised DCE-MRI image.
    """
    # Convert SimpleITK image to NumPy array
    image_np = sitk.GetArrayFromImage(image_sitk)
    
    # Apply Non-Local Means denoising
    denoised_np = restoration.denoise_nl_means(
        image_np,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=True
    )
    
    # Convert denoised NumPy array back to SimpleITK image
    denoised_image_sitk = sitk.GetImageFromArray(denoised_np)
    denoised_image_sitk.CopyInformation(image_sitk)  # Preserve original metadata
    
    return denoised_image_sitk

def clip_image_sitk(image_sitk, percentiles=[1, 99]):
    """Clip intensity range of an image.

    Parameters
    image: ITK Image
        Image to normalize
    lowerbound: float, default -1000.0
        lower bound of clipping range
    upperbound: float, default 3000.0
        lower bound of clipping range

    Returns
    -------
    image : ITK Image
        Output image.

    """
    image_array = sitk.GetArrayFromImage(image_sitk)
    image_array = image_array.ravel()
    # Drop all zeroes from array
    image_array = image_array[image_array != 0]
    lowerbound = np.percentile(image_array, percentiles[0])
    upperbound = np.percentile(image_array, percentiles[1])
    # Create clamping filter for clipping and set variables
    filter = sitk.ClampImageFilter()
    filter.SetLowerBound(float(lowerbound))
    filter.SetUpperBound(float(upperbound))

    # Execute
    clipped_image_sitk = filter.Execute(image_sitk)

    return clipped_image_sitk

def zscore_normalization_sitk(image_sitk, mean, std):
    # Z-score normalization
    array = sitk.GetArrayFromImage(image_sitk) 
    normalized_array = (array - mean) / std
    zscored_sitk = sitk.GetImageFromArray(normalized_array)
    zscored_sitk.CopyInformation(image_sitk)
    return zscored_sitk

def histogram_equalization_sitk(image_sitk, alpha=0.3, beta=0.3, radius=5):
    """Perform histogram equalization on an image.
    
    See for documentation https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1AdaptiveHistogramEqualizationImageFilter.html.
    
     Parameters
    ----------
    image_sitk: ITK Image
        Image to normalize
        
    alpha: float, default 0.3
        controls how much the filter acts like the classical histogram equalization
        method (alpha=0) to how much the filter acts like an unsharp mask (alpha=1).
        
    beta: float, default 0.3
        controls how much the filter acts like an unsharp mask (beta=0) to much
        the filter acts like pass through (beta=1, with alpha=1).
    
    radius: integer, default None
       Controls the size of the region over which local statistics are calculated.
       The size of the window is controlled by SetRadius the default Radius is 5 in all directions.

    Returns
    -------
    image : ITK Image
        Output image.
        
    """
    hqfilter = sitk.AdaptiveHistogramEqualizationImageFilter()
    hqfilter.SetAlpha(alpha)
    hqfilter.SetBeta(beta)
    hqfilter.SetRadius(radius) 
    heq_image = hqfilter.Execute(image_sitk)
    return heq_image

def resample_sitk(image_sitk, new_spacing=None, new_size=None,
                   interpolator=sitk.sitkBSpline, tol=0.00001):
    # Get original settings
    original_size = image_sitk.GetSize()
    original_spacing = image_sitk.GetSpacing()
   
    # ITK can only do 3D images
    if len(original_size) == 2:
        original_size = original_size + (1, )
    if len(original_spacing) == 2:
        original_spacing = original_spacing + (1.0, )

    if new_size is None:
        # Compute output size
        new_size = [round(original_size[0]*(original_spacing[0] + tol) / new_spacing[0]),
                    round(original_size[1]*(original_spacing[0] + tol) / new_spacing[1]),
                    round(original_size[2]*(original_spacing[2] + tol) / new_spacing[2])]

    if new_spacing is None:
        # Compute output spacing
        tol = 0
        new_spacing = [original_size[0]*(original_spacing[0] + tol)/new_size[0],
                       original_size[1]*(original_spacing[0] + tol)/new_size[1],
                       original_size[2]*(original_spacing[2] + tol)/new_size[2]]

    # Set and execute the filter
    ResampleFilter = sitk.ResampleImageFilter()
    ResampleFilter.SetInterpolator(interpolator)
    ResampleFilter.SetOutputSpacing(new_spacing)
    ResampleFilter.SetSize(np.array(new_size, dtype='int').tolist())
    ResampleFilter.SetOutputDirection(image_sitk.GetDirection())
    ResampleFilter.SetOutputOrigin(image_sitk.GetOrigin())
    ResampleFilter.SetOutputPixelType(image_sitk.GetPixelID())
    ResampleFilter.SetTransform(sitk.Transform())
    try:
        resampled_image_sitk = ResampleFilter.Execute(image_sitk)
    except RuntimeError:
        # Assume the error is due to the direction determinant being 0
        # Solution: simply set a correct direction
        # print('Bad output direction in resampling, resetting direction.')
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ResampleFilter.SetOutputDirection(direction)
        image_sitk.SetDirection(direction)
        resampled_image_sitk = ResampleFilter.Execute(image_sitk)

    return resampled_image_sitk