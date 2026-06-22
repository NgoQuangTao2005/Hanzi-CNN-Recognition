import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
from deep_translator import GoogleTranslator

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
model.load_state_dict(torch.load('hanzi_best_weight.pth', map_location=device))
model.eval()

LANGUAGE_CODES = {
    "English 🇬🇧": "en",
    "Vietnamese 🇻🇳": "vi",
    "French 🇫🇷": "fr",
    "Russian 🇷🇺": "ru",
    "German 🇩🇪": "de",
    "Spanish 🇪🇸": "es",
    "Korean 🇰🇷": "ko",
    "Japanese 🇯🇵": "ja"
}

def letterbox_resize(img, target_size=(64, 64)):
    w, h = img.size
    max_dim = max(w, h)
    target_max = int(min(target_size) * 0.85) 
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
    links_html = "<h4>Search for details (Hán-Nôm Dictionary (for Vietnamese people) - open new tab):</h4><ul style='line-height: 1.8;'>"
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
            links_html += f"<li><a href='{full_url}' target='_blank' style='color: #2563eb; text-decoration: none; font-weight: bold;'>Lookup Link for 【 {hanzi_char} 】</a> - <i>{hsk_str}</i></li>"
            has_link = True
            
    if not has_link:
        links_html += "<li><i style='color: gray;'>The predicted characters are not in the HSK 1-6 range.</i></li>"
        
    links_html += "</ul>"
    
    return result_label, links_html

def predict_realtime_buttons(image):
    default_html = "<i>Search suggestions will appear here...</i>"
    if image is None: 
        return "？", "？", "？", default_html
    img = image.get("composite", image.get("background", None)) if isinstance(image, dict) else image
    if img is None: 
        return "？", "？", "？", default_html
    
    img_gray = img.convert('L')
    img_np = np.array(img_gray)
    
    if np.all(img_np >= 250) or np.all(img_np == 0):
        return "？", "？", "？", default_html
        
    res_dict, links_html = process_and_predict(img, use_auto_crop=False)
    
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

def select_and_clear(selected_char, current_text):
    if selected_char == "？" or not selected_char:
        return current_text, gr.update()
    if not current_text:
        current_text = ""
    new_text = current_text + selected_char
    
    return new_text, {"background": None, "layers": [], "composite": None}

def translate_text(text, target_lang):
    """Translate Chinese text to one selected language"""
    if not text or text.strip() == "":
        return "Translation will appear here..."
    try:
        if target_lang in LANGUAGE_CODES:
            target_lang_code = LANGUAGE_CODES[target_lang]
           
            result = GoogleTranslator(source='zh-CN', target=target_lang_code).translate(text)
            return result
        else:
            return "Selected language not supported"
    except Exception as e:
        return f"Translation Error: {str(e)}"
    
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Real-time Chinese Handwriting Recognition & Translator</h1>")
    
    with gr.Tabs():
       
        with gr.TabItem("Character Recognition"):
            
            with gr.Row():
                with gr.Column(scale=1):
                    canvas = gr.Sketchpad(type="pil", label="Write Chinese character here", brush=gr.Brush(colors=["#000000"]))
                            
                with gr.Column(scale=1):
                    gr.Markdown("### Quick Search Results (Click to Select):")
                    with gr.Row():
                        btn_top1 = gr.Button("？", variant="primary", size="lg")
                        btn_top2 = gr.Button("？", variant="secondary", size="lg")
                        btn_top3 = gr.Button("？", variant="secondary", size="lg")
                    
                    output_links = gr.HTML(label="Search Links", value="<i>Search suggestions will appear here...</i>")
                    
            gr.Markdown("---")
            with gr.Row():
                output_text_area = gr.Textbox(
                    label="Complete Sentence/Paragraph (Can be copied or edited directly)", 
                    placeholder="The selected characters will be automatically appended here...",
                    lines=3,
                    scale=4
                )
                btn_clear_text = gr.Button("Clear Text", variant="stop", scale=1)
                    
            canvas.change(
                fn=predict_realtime_buttons, 
                inputs=canvas, 
                outputs=[btn_top1, btn_top2, btn_top3, output_links],
                queue=False
            )
            
      
        with gr.TabItem("Translator"):
            gr.Markdown("### Translate Chinese text to a language of your choice")
            
            with gr.Row():
                input_text = gr.Textbox(
                    label="Chinese Text Source",
                    placeholder="Text from Tab 1 will sync here automatically, or type manually...",
                    lines=4
                )
            
            with gr.Row():
               
                lang_dropdown = gr.Dropdown(
                    choices=list(LANGUAGE_CODES.keys()),
                    value="English 🇬🇧",
                    label="Target Language"
                )
                translate_btn = gr.Button("Translate", variant="primary", size="lg")
            
            gr.Markdown("---")
            
            with gr.Row():
                output_text = gr.Textbox(
                    label="Translation Result", 
                    value="Translation will appear here...", 
                    interactive=False, 
                    lines=5
                )

   
    def sync_tabs(text):
        return text

    output_text_area.change(
        fn=sync_tabs,
        inputs=output_text_area,
        outputs=input_text
    )

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
    
    btn_clear_text.click(fn=lambda: "", outputs=output_text_area)
    
    translate_btn.click(
        fn=translate_text,
        inputs=[input_text, lang_dropdown],
        outputs=output_text
    )
    lang_dropdown.change(
        fn=translate_text,
        inputs=[input_text, lang_dropdown],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
