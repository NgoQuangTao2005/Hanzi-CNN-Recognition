import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os

# ==========================================
# 1. KIẾN TRÚC MẠNG CNN 
# ==========================================
class HanziCNN(nn.Module):
    def __init__(self, num_classes):
        super(HanziCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # AdaptiveAvgPool2d thích ứng với kích thước đầu vào, đảm bảo đầu ra luôn là (256, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ==========================================
# 2. KHỞI TẠO MÔI TRƯỜNG & NẠP TỪ ĐIỂN
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('mapping.json', 'r', encoding='utf-8') as f:
    idx_to_hanzi = json.load(f)

with open('mapping_pinyin.json', 'r', encoding='utf-8') as f:
    idx_to_pinyin = json.load(f)

if os.path.exists('mapping_hsk.json'):
    with open('mapping_hsk.json', 'r', encoding='utf-8') as f:
        idx_to_hsk = json.load(f)
else:
    idx_to_hsk = {k: "HSK ?" for k in idx_to_hanzi.keys()}

NUM_CLASSES = len(idx_to_hanzi)
model = HanziCNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load('Tao_hanzi_best_weight.pth', map_location=device))
model.eval()

# ==========================================
# 3. CÁC HÀM XỬ LÝ ẢNH 
# ==========================================
def letterbox_resize(img, target_size=(64, 64)):
    """Hàm chuẩn hóa tỷ lệ y hệt lúc huấn luyện"""
    w, h = img.size
    max_dim = max(w, h)
    target_max = int(min(target_size) * 0.85) # Padding 85% an toàn
    ratio = target_max / max_dim
    new_w, new_h = int(w * ratio), int(h * ratio)
    
    resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
    img_resized = img.resize((new_w, new_h), resample_filter)
    
    new_img = Image.new('L', target_size, 255)
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img

def process_and_predict(img, use_auto_crop=False):
    # Nền trắng
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert('RGB')

    img_gray = img.convert('L')

    if use_auto_crop:
        img_arr = np.array(img_gray)
        binary_arr = np.where(img_arr < 200, 0, 255).astype(np.uint8)
        coords = np.argwhere(binary_arr == 0)
        
        if len(coords) > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            img_gray = img_gray.crop((x0, y0, x1, y1))

    img_64 = letterbox_resize(img_gray, target_size=(64, 64))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img_tensor = transform(img_64).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, 3)
        
    result_label = {}
    BASE_URL = "https://hvdic.thivien.net/whv/" 
    links_html = "<h4>Tra cứu chi tiết (Từ điển Hán Nôm - mở tab mới):</h4><ul style='line-height: 1.8;'>"
    has_link = False

    for i in range(3):
        idx_str = str(top_idxs[i].item())
        hanzi_char = idx_to_hanzi.get(idx_str, "?")
        pinyin_str = idx_to_pinyin.get(idx_str, "?")
        hsk_str = idx_to_hsk.get(idx_str, "Ngoài HSK")
        prob = float(top_probs[i].item())
        
        display_label = f"【 {hanzi_char} 】 ({pinyin_str}) | {hsk_str}"
        result_label[display_label] = prob
        
        if "HSK" in hsk_str and "Ngoài" not in hsk_str:
            full_url = f"{BASE_URL}{hanzi_char}"
            links_html += f"<li><a href='{full_url}' target='_blank' style='color: #2563eb; text-decoration: none; font-weight: bold;'>Link tra cứu chữ 【 {hanzi_char} 】</a> - <i>{hsk_str}</i></li>"
            has_link = True
            
    if not has_link:
        links_html += "<li><i style='color: gray;'>Các chữ dự đoán không nằm trong nhóm HSK 1-6.</i></li>"
        
    links_html += "</ul>"
    
    return result_label, links_html

# Hàm vẽ tay
def predict_from_sketch(image):
    if image is None: return None, ""
    img = image.get("composite", image.get("background", None)) if isinstance(image, dict) else image
    if img is None: return None, ""
    return process_and_predict(img, use_auto_crop=False) 

# Hàm tải ảnh lên
def predict_from_upload(image):
    if image is None: return None, ""
    return process_and_predict(image, use_auto_crop=False)

# ==========================================
# 4. GIAO DIỆN 
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>App nhận diện chữ Hán</h1>")
    gr.Markdown("<p style='text-align: center;'>Tra cứu nhanh chóng: chữ Hán + Phiên âm pinyin + Cấp độ HSK + Từ điển Hán Nôm</p>")
    gr.Markdown("<p style='text-align: center;'>Lưu ý: Viết nét chữ dày để kết quả nhận diện chính xác hơn</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Vẽ tay trên Canvas"):
                    canvas = gr.Sketchpad(type="pil", label="Viết chữ Hán vào đây", brush=gr.Brush(colors=["#000000"]))
                    btn_draw = gr.Button("Phân tích nét vẽ", variant="primary")
                with gr.TabItem("Tải ảnh lên"):
                    upload = gr.Image(type="pil", image_mode="L", label="Ảnh chữ Hán nền trắng nét bút đen")
                    btn_upload = gr.Button("Phân tích ảnh", variant="primary")
                    
        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=3, label="Kết quả phân tích (Top 3)")
            output_links = gr.HTML(label="Link tra cứu", value="<i>Gợi ý tra cứu sẽ hiển thị ở đây...</i>")
            
    btn_draw.click(fn=predict_from_sketch, inputs=canvas, outputs=[output_label, output_links])
    btn_upload.click(fn=predict_from_upload, inputs=upload, outputs=[output_label, output_links])

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)