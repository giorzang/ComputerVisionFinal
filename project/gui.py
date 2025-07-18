import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from model_init import get_model

# ==== Cấu hình ====
MODEL_PATH = "model.pth"              # hoặc checkpoint.pth của bạn
TARGET_SIZE = (640, 360)               # (width, height)
BLUR_KERNEL = (55, 55)

# ==== Thiết bị & Load mô hình ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==== Hàm tiện ích ====
def smart_resize(img, target_size=TARGET_SIZE):
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def refine_mask(mask, kernel_size=3):
    b = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8, cv2.CV_32S)
    if num_labels <= 1:
        return np.zeros_like(mask)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    main = (labels == (np.argmax(sizes) + 1)).astype(np.uint8) * 255
    closed = cv2.morphologyEx(main, cv2.MORPH_CLOSE, kernel, iterations=2)
    smooth = cv2.GaussianBlur(closed, (kernel_size*2+1,)*2, 0)
    return (smooth.astype(np.float32) / 255.0)

# Transform ảnh đầu vào cho model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((TARGET_SIZE[1], TARGET_SIZE[0])),
    transforms.ToTensor(),
])

# ==== GUI với tkinter ====
class SegmentationApp:
    def __init__(self, root):
        self.root = root
        root.title("Realtime Segmentation & Background")
        self.bg_mode = tk.StringVar(value="None")
        self.custom_bg = None

        # Control frame
        cf = tk.Frame(root)
        cf.pack(padx=5, pady=5)

        ttk.Label(cf, text="Background:").grid(row=0, column=0, sticky="w")
        for i, mode in enumerate(["None", "Blur", "Custom"]):
            ttk.Radiobutton(cf, text=mode, variable=self.bg_mode, value=mode).grid(row=0, column=1+i, padx=5)

        self.btn_bg = ttk.Button(cf, text="Select Image", command=self.load_custom_bg)
        self.btn_bg.grid(row=0, column=4, padx=10)

        # Panels
        pf = tk.Frame(root)
        pf.pack()

        self.lbl_orig   = tk.Label(pf)
        self.lbl_mask   = tk.Label(pf)
        self.lbl_output = tk.Label(pf)

        self.lbl_orig.grid( row=0, column=0, padx=5, pady=5)
        self.lbl_mask.grid( row=0, column=1, padx=5, pady=5)
        self.lbl_output.grid(row=0, column=2, padx=5, pady=5)

        # OpenCV VideoCapture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_SIZE[1])

        self.update_frame()

    def load_custom_bg(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.png")])
        if path:
            img = cv2.imread(path)
            self.custom_bg = cv2.resize(img, TARGET_SIZE)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        canvas = smart_resize(frame, TARGET_SIZE)

        # Dự đoán mask
        inp = transform(canvas).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)['out']
            prob = F.softmax(out, dim=1)[0,1].cpu().numpy()
        mask = (prob > 0.5).astype(np.float32)
        mask_refined = refine_mask(mask)
        mask_3c = np.repeat(mask_refined[:,:,None], 3, axis=2)

        # Tạo mask hiển thị
        mask_vis = (mask_refined*255).astype(np.uint8)
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        # Tính output theo mode
        mode = self.bg_mode.get()
        if mode == "None":
            output = canvas.copy()
        elif mode == "Blur":
            blurred = cv2.GaussianBlur(canvas, BLUR_KERNEL, 0)
            output = np.where(mask_3c==1, canvas, blurred)
        elif mode == "Custom" and self.custom_bg is not None:
            output = np.where(mask_3c==1, canvas, self.custom_bg)
        else:
            output = canvas.copy()

        # Hiển thị lên GUI
        self.show_image(self.lbl_orig,   canvas)
        self.show_image(self.lbl_mask,   mask_vis)
        self.show_image(self.lbl_output, output)

        self.root.after(30, self.update_frame)

    def show_image(self, lbl, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb).resize((320, 180))
        imgtk = ImageTk.PhotoImage(pil)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
