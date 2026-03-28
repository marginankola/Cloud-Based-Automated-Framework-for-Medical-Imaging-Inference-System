from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from monai.networks.nets import UNet
from scipy import ndimage as ndi

class Simple3DClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
        
device = "cuda" if torch.cuda.is_available() else "cpu"
ID_TO_CATEGORY = {0: "Glioma", 1: "Meningioma", 2: "Metastatic"}

CLASSIFIER_MODEL_PATH = os.path.join(settings.BASE_DIR, 'best_classifier.pth')
classifier_model = Simple3DClassifier(num_classes=3).to(device)
classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
classifier_model.eval()

SEGMENTATION_MODEL_PATH = os.path.join(settings.BASE_DIR, 'best_model.pth')
segmenter_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=0
).to(device)
segmenter_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=device))
segmenter_model.eval()


def preprocess_nifti(img_data, target_size):
    img_data = (img_data - img_data.mean()) / (img_data.std() + 1e-8)

    if img_data.shape[0] == img_data.shape[1] and img_data.shape[2] != img_data.shape[0]:
        img_data = np.transpose(img_data, (2, 0, 1))

    img_tensor = torch.from_numpy(img_data).float().unsqueeze(0)
    img_tensor = F.interpolate(img_tensor[None], size=target_size, mode='trilinear', align_corners=False)
    return img_tensor.to(device)


def predict_classify_and_segment(nifti_path):
    img = nib.load(nifti_path)
    img_data = img.get_fdata().astype(np.float32)
    original_affine = img.affine
    original_shape = img_data.shape

    predicted_class_label = "Unknown"
    
    with torch.no_grad():
        classifier_input = preprocess_nifti(img_data.copy(), target_size=(128, 128, 128))
        output_logits = classifier_model(classifier_input)
        
        predicted_idx = torch.argmax(output_logits, dim=1).cpu().item()
        predicted_class_label = ID_TO_CATEGORY.get(predicted_idx, "Unknown")

        segmenter_input = preprocess_nifti(img_data.copy(), target_size=(64, 128, 128))

        output_seg = segmenter_model(segmenter_input)
        
        pred_mask_small = torch.argmax(output_seg, dim=1).squeeze().cpu().numpy()

    zoom_factors = (
        original_shape[0] / pred_mask_small.shape[0],
        original_shape[1] / pred_mask_small.shape[1],
        original_shape[2] / pred_mask_small.shape[2]
    )
    pred_mask_large = ndi.zoom(pred_mask_small, zoom_factors, order=0)

    MIN_VOXEL_COUNT = 100 
    labels, num_features = ndi.label(pred_mask_large)
    
    if num_features > 0:
        component_sizes = np.bincount(labels.ravel())[1:]
        large_enough_indices = np.where(component_sizes >= MIN_VOXEL_COUNT)[0]
        large_enough_labels = large_enough_indices + 1
        clean_volume = np.isin(labels, large_enough_labels).astype(np.int16)
    else:
        clean_volume = pred_mask_large.astype(np.int16)

    header = img.header.copy()
    header.set_data_dtype(np.int16)
    predicted_nifti = nib.Nifti1Image(clean_volume, original_affine, header)
    
    return predicted_nifti, predicted_class_label


def index(request):
    original_url = None
    mask_url = None
    predicted_class = None
    error_message = None

    if request.method == 'POST' and request.FILES.get('nifti_file'):
        try:
            uploaded_file = request.FILES['nifti_file']
            
            # 1. Validate File Extension
            if not (uploaded_file.name.endswith('.nii') or uploaded_file.name.endswith('.nii.gz')):
                raise ValueError("Invalid file format. Please upload a .nii or .nii.gz file.")

            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_file_path = fs.path(filename)

            # --- Save Original as Brain Model ---
            base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
            original_filename = f'original_{base_name}.nii'
            original_filepath = os.path.join(settings.MEDIA_ROOT, original_filename)
            
            try:
                original_nifti = nib.load(uploaded_file_path)
                nib.save(original_nifti, original_filepath)
                original_url = fs.url(original_filename)
            except Exception as e:
                raise ValueError(f"Failed to process the uploaded NIfTI file. It might be corrupted. Error: {str(e)}")

            # --- Run Prediction ---
            try:
                predicted_nifti_mask, predicted_class = predict_classify_and_segment(uploaded_file_path)
            except Exception as e:
                 raise RuntimeError(f"AI Model failed to analyze the scan. Error: {str(e)}")

            # --- Save Predicted Mask ---
            mask_filename = f'predicted_{base_name}.nii'
            mask_filepath = os.path.join(settings.MEDIA_ROOT, mask_filename)
            nib.save(predicted_nifti_mask, mask_filepath)
            mask_url = fs.url(mask_filename)

        except ValueError as ve:
            error_message = str(ve)
            # Clean up if possible (optional)
        except RuntimeError as re:
            error_message = str(re)
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"

    return render(request, 'predictor/index.html', {
        'original_url': original_url,
        'mask_url': mask_url,
        'predicted_class': predicted_class,
        'error_message': error_message
    })