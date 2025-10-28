!pip install -q opencv-python-headless streamlit

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os

# --------------------------------------------
# Streamlit Page Setup
# --------------------------------------------
st.set_page_config(page_title="Thyroid Nodule Segmentation", layout="wide")
st.title("🩺 Thyroid Nodule Segmentation using Deep Learning")
st.markdown("Upload a thyroid ultrasound image to detect and visualize **thyroid nodules** (in red overlay).")

# --------------------------------------------
# 1️⃣ Model Definition (Residual U-Net)
# --------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.res(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)

class UNetRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ResidualBlock(1, 64)
        self.e2 = ResidualBlock(64, 128)
        self.e3 = ResidualBlock(128, 256)
        self.e4 = ResidualBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.b = ResidualBlock(512, 1024)
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.d4 = ResidualBlock(1024, 512)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.d3 = ResidualBlock(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.d2 = ResidualBlock(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2);  self.d1 = ResidualBlock(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b  = self.b(self.pool(e4))
        d4 = self.u4(b); d4 = self.d4(torch.cat([d4, e4], dim=1))
        d3 = self.u3(d4); d3 = self.d3(torch.cat([d3, e3], dim=1))
        d2 = self.u2(d3); d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = self.u1(d2); d1 = self.d1(torch.cat([d1, e1], dim=1))
        return self.final(d1)

# --------------------------------------------
# 2️⃣ Post-Processing (remove large components)
# --------------------------------------------
def remove_large_components(binary_mask, max_area_frac=0.15):
    h, w = binary_mask.shape
    max_area = h * w * max_area_frac
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype("uint8"), connectivity=8)
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] <= max_area:
            out[labels == lab] = 1
    return out

# --------------------------------------------
# 3️⃣ Model Loading
# --------------------------------------------
@st.cache_resource
def load_model():
    # Path to the original full model file
    model_path = "data/Train/model_output/thyroid_nodule_segmentation.pth"

    # Initialize model (same architecture used in training)
    model = UNetRes().to("cpu")

    # Load full precision model
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# --------------------------------------------
# 4️⃣ Image Upload + Preprocessing
# --------------------------------------------
uploaded_file = st.file_uploader("Upload a thyroid ultrasound image (grayscale):", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    input_tensor = transform(image).unsqueeze(0)

    # --- Inference ---
    with torch.no_grad():
        logits = model(input_tensor)
        pred_mask = torch.sigmoid(logits).squeeze().cpu().numpy()

    # --- Threshold ---
    threshold = st.slider("Select threshold for mask display", 0.0, 1.0, 0.5, 0.01)
    binary_mask = (pred_mask > threshold).astype(np.uint8)

    # --- Post-Processing ---
    binary_mask_filtered = remove_large_components(binary_mask, max_area_frac=0.15)

    # --------------------------------------------
    # 5️⃣ Visualization (with Alpha Overlay)
    # --------------------------------------------
    st.header("🔍 Segmentation Result")

    # Convert to RGB for blending
    original_image_disp = image.convert("RGB")

    # Resize mask to match display size
    mask_resized = np.array(Image.fromarray(binary_mask_filtered * 255).resize(original_image_disp.size))

    # Create RGBA mask (red overlay)
    mask_rgba = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 4), dtype=np.uint8)
    mask_rgba[..., 0] = 255  # red
    mask_rgba[..., 3] = mask_resized  # alpha mask

    mask_image = Image.fromarray(mask_rgba, mode="RGBA")
    alpha = 0.35  # overlay transparency (0 = invisible, 1 = opaque)
    overlay_image = Image.alpha_composite(original_image_disp.convert("RGBA"), mask_image)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original_image_disp, caption="Original Image", use_container_width=True)
    with col2:
        st.image(mask_image, caption="Predicted Mask (Transparent Red)", use_container_width=True)
    with col3:
        st.image(overlay_image, caption=f"Overlay Result (alpha={alpha})", use_container_width=True)

    st.markdown(f"**Note:** Red regions indicate detected thyroid nodules (Transparency α = {alpha}).")
else:
    st.info("👆 Please upload a thyroid ultrasound image to begin.")
