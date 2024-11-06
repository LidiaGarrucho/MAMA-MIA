import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure
from src.preprocessing import *
import matplotlib.patches as patches
import ipywidgets as widgets
from ipywidgets import interact

def get_segmentation_bounding_box(mask_sitk, margin=0, max_size=None, label=1):
    """
    Extracts the exact bounding box of a segmentation mask, with optional margins.
    """
    # Set full size to either max_size or the actual image size if max_size is not provided
    if max_size:
        full_size = max_size
    else:
        full_size = mask_sitk.GetSize()

    # Use LabelShapeStatisticsImageFilter to get the bounding box of the segmented region
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask_sitk)
    
    if label not in label_shape_filter.GetLabels():
        raise ValueError(f"Label {label} not found in segmentation mask.")
    
    bbox_voi = label_shape_filter.GetBoundingBox(label)
    
    # Bounding box coordinates from LabelShapeStatisticsImageFilter
    x_min_bb, y_min_bb, z_min_bb = bbox_voi[0], bbox_voi[1], bbox_voi[2]
    x_max_bb = bbox_voi[0] + bbox_voi[3] - 1  # width in x direction
    y_max_bb = bbox_voi[1] + bbox_voi[4] - 1  # height in y direction
    z_max_bb = bbox_voi[2] + bbox_voi[5] - 1  # depth in z direction
    
    # Apply optional margin, making sure it doesn't exceed image bounds
    x_min = max(0, x_min_bb - margin)
    y_min = max(0, y_min_bb - margin)
    z_min = max(0, z_min_bb - margin)
    x_max = min(full_size[0] - 1, x_max_bb + margin)
    y_max = min(full_size[1] - 1, y_max_bb + margin)
    z_max = min(full_size[2] - 1, z_max_bb + margin)

    # Return bounding box as a list
    bbox = [x_min, y_min, z_min, x_max, y_max, z_max]
    return bbox

def plot_mri_and_segmentation(image_sitk, mask_sitk, patient_id, display_slices=None, bounding_box=None,
                             color='red', marker='-', line_thickness=1, resample=False):
    """Plots 2D views of an MRI image with segmentation contours and optional bounding box"""
    
    # Get image spacing and orientation
    input_scale = image_sitk.GetSpacing()
    orientation = get_image_orientation_from_direction(image_sitk)

        # Define slice locations for display
    if display_slices is None:
        # Default to the middle slice along each axis of the bounding box containing the mask
        if bounding_box is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = bounding_box
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = get_segmentation_bounding_box(mask_sitk)
        display_slices = [(x_min + x_max) // 2, (y_min + y_max) // 2, (z_min + z_max) // 2]

    if resample:
        # Resample both image and mask to isotropic spacing [1x1x1]
        image_sitk = resample_sitk(image_sitk, new_spacing=[1, 1, 1], interpolator=sitk.sitkBSpline)
        mask_sitk = resample_sitk(mask_sitk, new_spacing=[1, 1, 1], interpolator=sitk.sitkNearestNeighbor)
        # Adjust display slices according to original spacing
        display_slices = [int(display_slices[0] * input_scale[0]), 
                          int(display_slices[1] * input_scale[1]), 
                          int(display_slices[2] * input_scale[2])]

    # Convert images to NumPy arrays
    image_array = sitk.GetArrayFromImage(image_sitk)
    mask_array = sitk.GetArrayFromImage(sitk.Cast(mask_sitk, sitk.sitkFloat32))

    # Extract 2D slices for each view
    slice_xy = image_array[:, :, display_slices[0]]
    slice_xz = image_array[:, display_slices[1], :]
    slice_yz = image_array[display_slices[2], :, :]
    
    # Extract corresponding mask slices
    mask_xy = mask_array[:, :, display_slices[0]]
    mask_xz = mask_array[:, display_slices[1], :]
    mask_yz = mask_array[display_slices[2], :, :]

    # Find contours for each mask slice
    contours_xy = measure.find_contours(mask_xy, 0.5)
    contours_xz = measure.find_contours(mask_xz, 0.5)
    contours_yz = measure.find_contours(mask_yz, 0.5)

    # Plot 2D views with contours
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Patient ID: {patient_id}", fontsize=12)
    
    ax[0].imshow(slice_xy, cmap='gray')
    ax[1].imshow(slice_xz, cmap='gray')
    ax[2].imshow(slice_yz, cmap='gray')

    # Plot contours on each view
    for contour in contours_xy:
        ax[0].plot(contour[:, 1], contour[:, 0], color=color, linestyle=marker, linewidth=line_thickness)
    for contour in contours_xz:
        ax[1].plot(contour[:, 1], contour[:, 0], color=color, linestyle=marker, linewidth=line_thickness)
    for contour in contours_yz:
        ax[2].plot(contour[:, 1], contour[:, 0], color=color, linestyle=marker, linewidth=line_thickness)

    # Plot bounding box if provided
    if bounding_box is not None:
        x_min, y_min, z_min, x_max, y_max, z_max = bounding_box
        if resample:
            # Rescale the bounding box to the new spacing
            x_min, x_max = int(x_min * input_scale[0] + 0.5), int(x_max * input_scale[0] + 0.5)
            y_min, y_max = int(y_min * input_scale[1] + 0.5), int(y_max * input_scale[1] + 0.5)
            z_min, z_max = int(z_min * input_scale[2] + 0.5), int(z_max * input_scale[2] + 0.5)
        # Define bounding box for each 2D view
        rect_xy = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='g', facecolor='none')
        rect_xz = patches.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min, linewidth=1, edgecolor='g', facecolor='none')
        rect_yz = patches.Rectangle((y_min, z_min), y_max - y_min, z_max - z_min, linewidth=1, edgecolor='g', facecolor='none')
        # Add bounding box to each view
        ax[2].add_patch(rect_xy)
        ax[1].add_patch(rect_xz)
        ax[0].add_patch(rect_yz)

    # Set titles and invert y-axis for radiological view
    view_titles = ["Sagittal", "Coronal", "Axial"] if orientation == 'LAI' else ["Coronal", "Axial", "Sagittal"]
    for i, title in enumerate(view_titles):
        ax[i].set_title(title, fontweight="bold")
        ax[i].invert_yaxis()
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_aspect('equal')

    fig.tight_layout()
    plt.show()

def plot_3d_segmentation(mask_sitk):
    orientation = get_image_orientation_from_direction(mask_sitk)
    if orientation == 'LAI':
        mask_array = sitk.GetArrayFromImage(mask_sitk).transpose(1, 2, 0)
    else:
        mask_array = sitk.GetArrayFromImage(mask_sitk)
    color_list = ['#E15759']
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(mask_array, facecolors=color_list[0], edgecolor='k')
    plt.tight_layout()
    plt.show()

def plot_mri_preprocessing(image_sitk, preprocessed_image_sitk, patient_id,
                                  preprocessing_method='', figsize=(12, 8), display_slices=None):
    """
    Displays a plot of MRI slices across all three planes before and after preprocessing.
    
    Parameters:
        image_sitk (SimpleITK.Image): Original MRI image.
        preprocessed_image_sitk (SimpleITK.Image): Preprocessed MRI image.
        patient_id (str): Identifier for the patient.
        preprocessing_method (str): Description of the preprocessing method applied.
        figsize (tuple): Size of the figure.
        display_slices (tuple, optional): Indices for the slices to display for each plane (axial, sagittal, coronal).
    """
    
    orientation = get_image_orientation_from_direction(image_sitk)
    
    # Convert SimpleITK images to NumPy arrays
    image_np = sitk.GetArrayFromImage(image_sitk)
    preprocessed_image_np = sitk.GetArrayFromImage(preprocessed_image_sitk)
    
    # Get image dimensions (after preprocessing)
    z_max, y_max, x_max = preprocessed_image_np.shape
    original_z_max, original_y_max, original_x_max = image_np.shape
    
    # Use the minimum size to avoid out-of-range errors
    z_max = min(z_max, original_z_max)
    y_max = min(y_max, original_y_max)
    x_max = min(x_max, original_x_max)

    # If display_slices is not provided, use the middle slice for each axis
    if display_slices is None:
        display_slices = (z_max // 2, y_max // 2, x_max // 2)

    # Function to plot slices in all three planes based on orientation
    def plot_slices(slicer_1, slicer_2, slicer_3):
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f"Patient ID: {patient_id} - {preprocessing_method}", fontsize=14)

        # Define slice views for each plane before preprocessing, adjusting slices based on orientation
        if orientation == 'LAI':
            # 'LAI' orientation (axial, coronal, sagittal)
            views_before = [
                image_np[slicer_1, :, :],      # Axial
                image_np[:, :, slicer_2],    # Sagittal
                image_np[:, slicer_3, :]    # Coronal
            ]
            views_after = [
                preprocessed_image_np[slicer_1, :, :],      # Axial
                preprocessed_image_np[:, :, slicer_2],    # Sagittal
                preprocessed_image_np[:, slicer_3, :]    # Coronal
            ]
            plane_titles = ["Axial", "Sagittal", "Coronal"]
        else:
            # Default orientation (sagittal, axial, coronal)
            views_before = [
                image_np[slicer_1, :, :],     # Axial
                image_np[:, slicer_2, :],  # Sagittal
                image_np[:, :, slicer_3]    # Coronal
            ]
            views_after = [
                preprocessed_image_np[slicer_1, :, :],     # Axial
                preprocessed_image_np[:, slicer_2, :],  # Sagittal
                preprocessed_image_np[:, :, slicer_3]    # Coronal
            ]
            plane_titles = ["Sagittal", "Axial", "Coronal"]

        # Plot each view before preprocessing
        for i, (view, title) in enumerate(zip(views_before, plane_titles)):
            axes[0, i].imshow(view, cmap='gray', interpolation='none')
            axes[0, i].set_title(f'Before - {title}', fontsize=10)
            axes[0, i].axis('off')
            axes[0, i].invert_yaxis()

        # Plot each view after preprocessing
        for i, (view, title) in enumerate(zip(views_after, plane_titles)):
            axes[1, i].imshow(view, cmap='gray', interpolation='none')
            axes[1, i].set_title(f'After - {title}', fontsize=10)
            axes[1, i].axis('off')
            axes[1, i].invert_yaxis()

        plt.tight_layout()
        plt.show()

    # Use the provided display_slices or default to the middle of each axis
    slicer_1, slicer_2, slicer_3 = display_slices
    plot_slices(slicer_1, slicer_2, slicer_3)

def interactive_plot_mri_preprocessing(image_sitk, preprocessed_image_sitk, patient_id,
                                       preprocessing_method='', figsize=(12, 8)):
    """
    Displays an interactive plot of MRI slices across all three planes before and after preprocessing.

    Parameters:
        image_sitk (SimpleITK.Image): Original MRI image.
        preprocessed_image_sitk (SimpleITK.Image): Preprocessed MRI image.
        patient_id (str): Identifier for the patient.
        preprocessing_method (str): Description of the preprocessing method applied.
        figsize (tuple): Size of the figure.
    """
    
    orientation = get_image_orientation_from_direction(image_sitk)
    # Convert SimpleITK images to NumPy arrays
    image_np = sitk.GetArrayFromImage(image_sitk)
    preprocessed_image_np = sitk.GetArrayFromImage(preprocessed_image_sitk)
    
    # Get image dimensions (after preprocessing)
    z_max, y_max, x_max = preprocessed_image_np.shape
    original_z_max, original_y_max, original_x_max = image_np.shape
    # Use the minimum size to avoid out-of-range errors
    z_max = min(z_max, original_z_max)
    y_max = min(y_max, original_y_max)
    x_max = min(x_max, original_x_max)

    # Function to plot slices in all three planes based on orientation
    def plot_slices(slicer_1, slicer_2, slicer_3):
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f"Patient ID: {patient_id} - {preprocessing_method}", fontsize=14)

        # Define slice views for each plane before preprocessing, adjusting slices based on orientation
        if orientation == 'LAI':
            # 'LAI' orientation (axial, coronal, sagittal)
            views_before = [
                image_np[slicer_1, :, :],      # Axial
                image_np[:, :, slicer_2],    # Sagittal
                image_np[:, slicer_3, :]    # Coronal
            ]
            views_after = [
                preprocessed_image_np[slicer_1, :, :],      # Axial
                preprocessed_image_np[:, :, slicer_2],    # Sagittal
                preprocessed_image_np[:, slicer_3, :]    # Coronal
                
            ]
            plane_titles = ["Axial", "Sagittal", "Coronal"]
        else:
            # Default orientation (sagittal, axial, coronal)
            views_before = [
                image_np[slicer_1, :, :],     # Axial
                image_np[:, slicer_2, :],  # Sagittal
                image_np[:, :, slicer_3]    # Coronal
            ]
            views_after = [
                preprocessed_image_np[slicer_1, :, :],     # Axial
                preprocessed_image_np[:, slicer_2, :],  # Sagittal
                preprocessed_image_np[:, :, slicer_3]    # Coronal
            ]
            plane_titles = ["Sagittal", "Axial", "Coronal"]

        # Plot each view before preprocessing
        for i, (view, title) in enumerate(zip(views_before, plane_titles)):
            axes[0, i].imshow(view, cmap='gray', interpolation='none')
            axes[0, i].set_title(f'Before - {title}', fontsize=10)
            axes[0, i].axis('off')
            axes[0, i].invert_yaxis()

        # Plot each view after preprocessing
        for i, (view, title) in enumerate(zip(views_after, plane_titles)):
            axes[1, i].imshow(view, cmap='gray', interpolation='none')
            axes[1, i].set_title(f'After - {title}', fontsize=10)
            axes[1, i].axis('off')
            axes[1, i].invert_yaxis()

        plt.tight_layout()
        plt.show()

    # Interactive sliders for choosing slice indices based on orientation
    if orientation == 'LAI':
        # 'LAI' orientation (axial, coronal, sagittal)
        interact(plot_slices,
                 slicer_1=widgets.IntSlider(min=0, max=z_max - 1, step=1, value=z_max // 2),
                 slicer_2=widgets.IntSlider(min=0, max=y_max - 1, step=1, value=y_max // 2),
                 slicer_3=widgets.IntSlider(min=0, max=x_max - 1, step=1, value=x_max // 2))
    else:
        # Default orientation (sagittal, axial, coronal)
        interact(plot_slices,
                 slicer_2=widgets.IntSlider(min=0, max=y_max - 1, step=1, value=y_max // 2),
                 slicer_1=widgets.IntSlider(min=0, max=z_max - 1, step=1, value=z_max // 2),
                 slicer_3=widgets.IntSlider(min=0, max=x_max - 1, step=1, value=x_max // 2))

def interactive_plot_mri_and_segmentation(image_sitk, mask_sitk, patient_id, resample=False,
                                          color='red', marker='-', line_thickness=1, figsize=(15, 5)):
    """
    Displays interactive 2D slices of an MRI image with segmentation contours.

    Parameters:
        image_sitk (SimpleITK.Image): Input MRI image.
        mask_sitk (SimpleITK.Image): Segmentation mask.
        patient_id (str): Identifier for the patient.
        color (str, optional): Color of segmentation contours.
        marker (str, optional): Line style for contours.
        line_thickness (int, optional): Thickness of contour lines.
        figsize (tuple, optional): Size of the figure for the plot.
    """
    if resample:
        # Resample both image and mask to isotropic spacing [1x1x1]
        image_sitk = resample_sitk(image_sitk, new_spacing=[1, 1, 1], interpolator=sitk.sitkBSpline)
        mask_sitk = resample_sitk(mask_sitk, new_spacing=[1, 1, 1], interpolator=sitk.sitkNearestNeighbor)

    # Convert SimpleITK images to NumPy arrays
    image_array = sitk.GetArrayFromImage(image_sitk)
    mask_array = sitk.GetArrayFromImage(sitk.Cast(mask_sitk, sitk.sitkFloat32))

    # Get image spacing and orientation
    input_scale = image_sitk.GetSpacing()
    orientation = get_image_orientation_from_direction(image_sitk)

    # Get dimensions after resampling
    size_x, size_y, size_z = image_array.shape

    # Function to update the plot based on selected slice indices for each axis
    def update_plot(slicer_1, slicer_2, slicer_3):
        # Adjust slice indices based on the orientation of the image
        if orientation == 'LAI':
            # 'LAI' orientation (axial, coronal, sagittal)
            slice_idx_xy = min(slicer_1, size_z - 1)  # Axial view (along Z-axis)
            slice_idx_xz = min(slicer_3, size_y - 1)  # Coronal view (along Y-axis)
            slice_idx_yz = min(slicer_2, size_x - 1)  # Sagittal view (along X-axis)
        else:
            # Default orientation (sagittal, axial, coronal)
            slice_idx_xy = min(slicer_2, size_z - 1)  # Sagittal view (along Z-axis)
            slice_idx_xz = min(slicer_1, size_y - 1)  # Axial view (along Y-axis)
            slice_idx_yz = min(slicer_3, size_x - 1)  # Coronal view (along X-axis)

        if orientation == 'LAI':
            # Extract slices for each view
            slice_xy = image_array[:, :, slice_idx_xy]
            slice_xz = image_array[slice_idx_yz, :, :]
            slice_yz = image_array[:, slice_idx_xz, :]

            # Extract corresponding mask slices
            mask_xy = mask_array[:, :, slice_idx_xy]
            mask_xz = mask_array[slice_idx_yz, :, :]
            mask_yz = mask_array[:, slice_idx_xz, :]
        else:
            # Extract slices for each view
            slice_xy = image_array[:, slice_idx_xz, :]
            slice_xz = image_array[slice_idx_yz, :, :]   
            slice_yz = image_array[:, :, slice_idx_xy] 

            # Extract corresponding mask slices
            mask_xy = mask_array[:, slice_idx_xz, :]
            mask_xz = mask_array[slice_idx_yz, :, :]
            mask_yz = mask_array[:, :, slice_idx_xy]

        # Find contours for each mask slice
        contours_xy = measure.find_contours(mask_xy, 0.5)
        contours_xz = measure.find_contours(mask_xz, 0.5)
        contours_yz = measure.find_contours(mask_yz, 0.5)

        # Plot 2D views with contours
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f"Patient ID: {patient_id}", fontsize=12)

        # Plot the slices
        ax[0].imshow(slice_xy, cmap='gray')
        ax[1].imshow(slice_xz, cmap='gray')
        ax[2].imshow(slice_yz, cmap='gray')

        # Plot contours for each view
        for contour in contours_xy:
            ax[0].plot(contour[:, 1], contour[:, 0], color=color, linestyle=marker, linewidth=line_thickness)
        for contour in contours_xz:
            ax[1].plot(contour[:, 1], contour[:, 0], color=color, linestyle=marker, linewidth=line_thickness)
        for contour in contours_yz:
            ax[2].plot(contour[:, 1], contour[:, 0], color=color, linestyle=marker, linewidth=line_thickness)

        # Set titles and invert y-axis for radiological view
        view_titles = ["Sagittal", "Axial", "Coronal"] if orientation == 'LAI' else ["Axial", "Sagittal", "Coronal"]
        for i, title in enumerate(view_titles):
            ax[i].set_title(title, fontweight="bold")
            ax[i].invert_yaxis()
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_aspect('equal')

        fig.tight_layout()
        plt.show()
        plt.close(fig)

    # Create interactive sliders for axial, sagittal, and coronal slices
    if orientation == 'LAI':
        interact(update_plot,
                    slicer_2=widgets.IntSlider(min=0, max=size_x - 1, step=1, value=size_x // 2),
                    slicer_1=widgets.IntSlider(min=0, max=size_z - 1, step=1, value=size_z // 2),
                    slicer_3=widgets.IntSlider(min=0, max=size_y - 1, step=1, value=size_y // 2))
    else:
        # Default orientation (sagittal, axial, coronal)
        interact(update_plot,
                 slicer_1=widgets.IntSlider(min=0, max=size_z - 1, step=1, value=size_z // 2),
                 slicer_2=widgets.IntSlider(min=0, max=size_y - 1, step=1, value=size_y // 2),
                 slicer_3=widgets.IntSlider(min=0, max=size_x - 1, step=1, value=size_x // 2))