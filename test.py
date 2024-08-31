from huggingface_hub import hf_hub_download
import torch
from PIL import Image
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import streamlit as st
import os
from datetime import datetime
from fastai.vision.learner import create_body
from torchvision.models import resnet34
from fastai.vision.models.unet import DynamicUnet
from with_attachments import send_emails

#Download the model from Hugging Face Hub 
repo_id = "Hammad712/GAN-Colorization-Model"
model_filename = "generator.pt"
model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)

#Define the generator model (same architecture as used during training)
def build_generator(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = create_body(resnet34(), pretrained=True, n_in=n_input, cut=-2)
    G_net = DynamicUnet(backbone, n_output, (size, size)).to(device)
    return G_net

@st.cache_resource
def load_model():
    repo_id = "Hammad712/GAN-Colorization-Model"
    model_filename = "generator.pt"
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_net = build_generator(n_input=1, n_output=2, size=256)
    G_net.load_state_dict(torch.load(model_path, map_location=device))
    G_net.eval()
    return G_net

#Preprocessing function 
def preprocess_image(img_path): 
    img = Image.open(img_path).convert("RGB")
    img = transforms.Resize((256, 256), Image.BICUBIC)(img) 
    img = np.array(img) 
    img_to_lab = rgb2lab(img).astype("float32")
    img_to_lab = transforms.ToTensor()(img_to_lab) 
    L = img_to_lab[[0], ...] / 50. - 1. 
    return L.unsqueeze(0).to(device)

#Inference function 
def colorize_image(img_path, model): 
    L = preprocess_image(img_path) 
    with torch.no_grad(): 
        ab = model(L) 
        L = (L + 1.) * 50. 
        ab = ab * 110. 
        Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy() 
        rgb_imgs = [] 
        for img in Lab: 
            img_rgb = lab2rgb(img)
            rgb_imgs.append(img_rgb) 
            return np.stack(rgb_imgs, axis=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a folder to store uploaded images
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("SIH 1733 - Team Helix")
st.title("Image Colorization App")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
user_email = st.text_input("Enter your email address:")

uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

if st.button("Colorize Image"):
    if not user_email:
        st.error("Please enter your email address.")
    elif not uploaded_file:
        st.error("Please upload an image before colorizing.")
    else:
        # Save the uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(uploaded_file.name)[1]
        saved_filename = f"{timestamp}{file_extension}"
        saved_path = os.path.join(UPLOAD_FOLDER, saved_filename)
        
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Colorizing..."):
            input_image = Image.open(saved_path).convert("RGB")
            
            model = load_model()
            colorized_image = colorize_image(saved_path, model)
            
            # Save the colorized image
            colorized_filename = f"colorized_{saved_filename}"
            colorized_path = os.path.join(UPLOAD_FOLDER, colorized_filename)
            Image.fromarray((colorized_image[0] * 255).astype(np.uint8)).save(colorized_path)
            
            st.image(colorized_image[0], caption="Colorized Image", use_column_width=True)
            
            # Send email with both the original and colorized images
            send_emails([user_email], [saved_path, colorized_path])
            st.success("Original and colorized images have been sent to your email!")

# Set device globally