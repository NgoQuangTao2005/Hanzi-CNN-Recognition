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

# Hàm dự đoán thời gian thực để map nhãn chữ đơn lẻ lên Nút bấm
def predict_realtime_buttons(image):
    default_html = "<i>Gợi ý tra cứu sẽ hiển thị ở đây...</i>"
    if image is None: 
        return "？", "？", "？", default_html
    img = image.get("composite", image.get("background", None)) if isinstance(image, dict) else image
    if img is None: 
        return "？", "？", "？", default_html
    
    # Chuyển đổi sang ảnh xám để kiểm tra
    img_gray = img.convert('L')
    img_np = np.array(img_gray)
    
    # --- ĐOẠN THÊM VÀO ĐỂ BẮT ẢNH TRẮNG ---
    # Nếu ảnh trắng tinh (không có nét vẽ nào hoặc là ảnh reset) thì trả về trạng thái trống
    if np.all(img_np >= 250) or np.all(img_np == 0):
        return "？", "？", "？", default_html
    # --------------------------------------
        
    res_dict, links_html = process_and_predict(img, use_auto_crop=False)
    
    # Bóc tách ký tự chữ Hán nằm trong dấu 【 】 từ key của từ điển trả về
    hanzi_outputs = []
    for key in res_dict.keys():
        try:
            char = key.split("【")[1].split("】")[0].strip()
            hanzi_outputs.append(char)
        except:
            hanzi_outputs.append("？")
            
    while len(hanzi_outputs) < 3:
        hanzi_outputs.append("？")
        
    return hanzi_outputs[0], hanzi_outputs[1], hanzi_outputs[2], links_html

# Hàm cộng dồn chữ vào đoạn văn tổng và kích hoạt xóa bảng vẽ Canvas
def select_and_clear(selected_char, current_text):
    if selected_char == "？" or not selected_char:
        return current_text, gr.update()
    if not current_text:
        current_text = ""
    new_text = current_text + selected_char
    
    # Sử dụng cấu trúc dictionary trống chuẩn của Gradio 4+ 
    # Giúp xóa sạch sành sanh mọi layer nét vẽ cũ mà KHÔNG làm reset độ dày cọ bút!
    return new_text, {"background": None, "layers": [], "composite": None}

# ==========================================
# 4. GIAO DIỆN BỘ GÕ CHỮ HÁN VIẾT TAY
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Hệ thống bộ gõ chữ Hán viết tay thời gian thực</h1>")
    gr.Markdown("<p style='text-align: center;'>Viết nét chữ -> Bấm chọn chữ đúng hệ thống gợi ý để tự động ghép thành câu văn</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Khung Canvas vẽ tay thời gian thực
            canvas = gr.Sketchpad(type="pil", label="Viết từng chữ vào đây", brush=gr.Brush(colors=["#000000"]))
                    
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Kết quả gợi ý nhanh (Bấm để chọn chữ):")
            with gr.Row():
                btn_top1 = gr.Button("？", variant="primary", size="lg")
                btn_top2 = gr.Button("？", variant="secondary", size="lg")
                btn_top3 = gr.Button("？", variant="secondary", size="lg")
            
            output_links = gr.HTML(label="Link tra cứu", value="<i>Gợi ý tra cứu sẽ hiển thị ở đây...</i>")
            
    # Ô chứa kết quả toàn bộ câu văn
    gr.Markdown("---")
    with gr.Row():
        output_text_area = gr.Textbox(
            label="📝 Câu/Đoạn văn bản hoàn chỉnh (Có thể trực tiếp sao chép hoặc sửa đổi)", 
            placeholder="Chữ ông chọn ở trên sẽ tự động nối đuôi xếp vào đây...",
            lines=3,
            scale=4
        )
        btn_clear_text = gr.Button("Xóa đoạn văn", variant="stop", scale=1)
            
    # --- THIẾT LẬP LOGIC TƯƠNG TÁC ---
    # 1. Tự động dự đoán và hiển thị nhãn chữ lên nút bấm khi viết
    canvas.change(
        fn=predict_realtime_buttons, 
        inputs=canvas, 
        outputs=[btn_top1, btn_top2, btn_top3, output_links],
        queue=False
    )
    
    # 2. Logic khi click chọn chữ: Ghép chữ vào Textbox và tự động xóa bảng vẽ để viết chữ tiếp theo
    btn_top1.click(
        fn=select_and_clear, 
        inputs=[btn_top1, output_text_area], 
        outputs=[output_text_area, canvas]
    )
    btn_top2.click(
        fn=select_and_clear, 
        inputs=[btn_top2, output_text_area], 
        outputs=[output_text_area, canvas]
    )
    btn_top3.click(
        fn=select_and_clear, 
        inputs=[btn_top3, output_text_area], 
        outputs=[output_text_area, canvas]
    )
    
    # Nút xóa nhanh toàn bộ chuỗi văn bản đã gõ
    btn_clear_text.click(fn=lambda: "", outputs=output_text_area)

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
