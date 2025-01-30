import requests
from pathlib import Path

def download_model():
    url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
    path = Path('models/SwinIR/swinir_4x.pth')
    path.parent.mkdir(exist_ok=True)
    
    if not path.exists():
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)
    return path

if __name__ == '__main__':
    download_model() 