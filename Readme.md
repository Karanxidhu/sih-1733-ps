# SAR Image Colorization App

This Streamlit app colorizes grayscale Synthetic Aperture Radar (SAR) images using a GAN-based model.

## Features

- Upload grayscale SAR images
- Colorize images using a pre-trained GAN model
- Send original and colorized images via email

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run test.py
   ```
2. Enter your email address
3. Upload a grayscale SAR image
4. Click "Colorize Image"
5. View the colorized image and receive both images via email

## Files

- `test.py`: Main Streamlit app
- `with_attachments.py`: Email sending functionality
- `requirements.txt`: Required Python packages

## Model

The app uses a pre-trained GAN model hosted on Hugging Face Hub:
- Repository: Hammad712/GAN-Colorization-Model
- Filename: generator.pt

## Note

Ensure you have proper credentials and permissions for the email functionality.
