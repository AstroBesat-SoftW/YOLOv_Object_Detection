import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math

# COCO Sınıflarının Tam Türkçe Çevirisi
CLASSES = [
    "insan", "bisiklet", "araba", "motosiklet", "uçak", "otobüs", "tren", "kamyon", "tekne", "trafik lambası",
    "yangın musluğu", "dur tabelası", "park metresi", "bank", "kuş", "kedi", "köpek", "at", "koyun", "inek",
    "fil", "ayı", "zebra", "zürafa", "sırt çantası", "şemsiye", "el çantası", "kravat", "valiz", "frizbi",
    "kayak", "kar tahtası", "spor topu", "uçurtma", "beyzbol sopası", "beyzbol eldiveni", "kaykay", "sörf tahtası",
    "tenis raketi", "şişe", "şarap kadehi", "fincan", "çatal", "bıçak", "kaşık", "kase", "muz", "elma", "sandviç",
    "portakal", "brokoli", "havuç", "sosisli sandviç", "pizza", "donut", "pasta", "sandalye", "kanepe", "saksı bitkisi",
    "yatak", "yemek masası", "tuvalet", "televizyon", "dizüstü bilgisayar", "fare", "kumanda", "klavye", "cep telefonu",
    "mikrodalga", "fırın", "ekmek kızartma makinesi", "lavabo", "buzdolabı", "kitap", "saat", "vazo", "makas",
    "oyuncak ayı", "saç kurutma makinesi", "diş fırçası"
]

# Paint ekranında göstermek için bazı emojiler (Görsel temsil için)
EMOJIS = {
    "insan": "🧍", "araba": "🚗", "şişe": "🍾", "cep telefonu": "📱", "kedi": "🐈", "köpek": "🐕",
    "dizüstü bilgisayar": "💻", "sandalye": "🪑", "kupa": "☕", "kitap": "📖", "fare": "🖱️"
}

class YoloApp:
    def __init__(self, root, window_title):
        self.root = root
        self.root.title(window_title)
        
        self.cap = cv2.VideoCapture(1)
        
        self.net = cv2.dnn.readNet("yolov3_last.weights", "yolov3.cfg")
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.otonom_mod = tk.BooleanVar(value=False)
        self.target_class = tk.StringVar(value="Hepsi")

        self.setup_gui()
        self.update_frame()

    def setup_gui(self):
        # 1. SOL EKRAN: Kamera
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(self.video_frame, text="Canlı Kamera", font=("Arial", 12, "bold")).pack()
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="black")
        self.canvas.pack()

        # 2. ORTA EKRAN: Paint (Radar/Harita)
        self.paint_frame = tk.Frame(self.root)
        self.paint_frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(self.paint_frame, text="Radar & Sinir Ağı Haritası", font=("Arial", 12, "bold")).pack()
        self.paint_canvas = tk.Canvas(self.paint_frame, width=640, height=480, bg="white", highlightbackground="gray", highlightthickness=2)
        self.paint_canvas.pack()

        # 3. SAĞ EKRAN: Kontrol Paneli
        self.control_frame = tk.Frame(self.root, width=250)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)

        tk.Label(self.control_frame, text="IoT Kontrol Paneli", font=("Arial", 16, "bold")).pack(pady=10)

        tk.Label(self.control_frame, text="Algılama Hassasiyeti:").pack(pady=5)
        self.slider = ttk.Scale(self.control_frame, from_=0.1, to=1.0, variable=self.conf_threshold, orient=tk.HORIZONTAL)
        self.slider.pack(fill=tk.X, pady=5)

        tk.Frame(self.control_frame, height=2, bg="gray").pack(fill=tk.X, pady=15)

        self.btn_otonom = tk.Checkbutton(self.control_frame, text="🤖 Otonom Takibi Başlat", 
                                       variable=self.otonom_mod, font=("Arial", 12, "bold"), fg="blue")
        self.btn_otonom.pack(pady=10)

        tk.Frame(self.control_frame, height=2, bg="gray").pack(fill=tk.X, pady=15)

        tk.Label(self.control_frame, text="Sadece Bunu Takip Et:", font=("Arial", 12, "bold")).pack(pady=5)
        
        options = ["Hepsi", "insan", "şişe", "cep telefonu", "araba", "kedi", "köpek"]
        for opt in options:
            ttk.Radiobutton(self.control_frame, text=opt.capitalize(), variable=self.target_class, value=opt).pack(anchor=tk.W, pady=2)

    def estimate_distance(self, frame_height, bbox_height):
        """Kutu yüksekliğine göre tahmini bir mesafe formülü uyduruyoruz."""
        if bbox_height == 0: return 0
        # Referans: Eğer ekran boyunun yarısı kadar (240px) yer kaplıyorsa ~1.5 metre diyelim
        # Ters orantı: Mesafe = Sabit / BBox_Yüksekliği
        distance = (frame_height / bbox_height) * 0.8
        return round(distance, 1)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            height, width, channels = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids, confidences, boxes = [], [], []
            
            # Bu frame'de algılanan nesnelerin Paint için verileri: (x, y, label, mesafe, color)
            detected_objects = []

            target_cx = None 
            target_found = False

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.conf_threshold.get():
                        detected_name = CLASSES[class_id]
                        selected_target = self.target_class.get()

                        if selected_target == "Hepsi" or detected_name == selected_target:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                            if not target_found and (selected_target == "Hepsi" or detected_name == selected_target):
                                target_cx = center_x
                                target_found = True

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold.get(), 0.4)

            # Paint ekranını her karede temizle
            self.paint_canvas.delete("all")

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(CLASSES[class_ids[i]])
                    color = self.colors[class_ids[i]]
                    
                    # Kamerada Çizim
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Paint verilerini toparla
                    cx = x + w // 2
                    cy = y + h // 2
                    dist = self.estimate_distance(height, h)
                    
                    # Rengi hex formatına çevir (Tkinter Paint canvas için)
                    hex_color = '#%02x%02x%02x' % (int(color[2]), int(color[1]), int(color[0]))
                    detected_objects.append((cx, cy, label, dist, hex_color))

            # --- PAINT EKRANINA SİNİR AĞI VE OBJELERİ ÇİZME ---
            
            # Önce ağ (network) çizgilerini çiz (noktalar arası)
            for i in range(len(detected_objects)):
                for j in range(i + 1, len(detected_objects)):
                    p1_x, p1_y = detected_objects[i][0], detected_objects[i][1]
                    p2_x, p2_y = detected_objects[j][0], detected_objects[j][1]
                    
                    # Aralarındaki mesafe piksel cinsinden (Oklar için hesap)
                    pixel_dist = math.hypot(p2_x - p1_x, p2_y - p1_y)
                    
                    # Çizgi çek (Ortasına aralarındaki uzaklık bağı yazılabilir)
                    self.paint_canvas.create_line(p1_x, p1_y, p2_x, p2_y, fill="gray", dash=(4, 4), width=1)

            # Sonra objelerin kendisini ve uzaklık oklarını çiz (çizgilerin üstünde dursun diye sonra çizdiriyoruz)
            for obj in detected_objects:
                cx, cy, label, dist, hex_color = obj
                emoji = EMOJIS.get(label, "📦") # Sözlükte yoksa kutu emojisi koy
                
                # Hedefi işaretleyen bir radar halkası
                self.paint_canvas.create_oval(cx-20, cy-20, cx+20, cy+20, outline=hex_color, width=2)
                self.paint_canvas.create_oval(cx-25, cy-25, cx+25, cy+25, outline=hex_color, dash=(2, 2))
                
                # Emoji ve Yazı (Resim niyetine)
                self.paint_canvas.create_text(cx, cy, text=emoji, font=("Arial", 20))
                self.paint_canvas.create_text(cx, cy + 35, text=f"{label.upper()}", font=("Arial", 10, "bold"), fill=hex_color)
                
                # Kameradan uzaklık tahmini (ok simgesiyle)
                self.paint_canvas.create_text(cx, cy + 50, text=f"↕ {dist}m", font=("Arial", 9), fill="black")

            # --- OTONOM MOTOR KONTROLÜ ---
            if self.otonom_mod.get():
                left_zone = width // 3
                right_zone = 2 * (width // 3)
                
                cv2.line(frame, (left_zone, 0), (left_zone, height), (255, 255, 255), 1)
                cv2.line(frame, (right_zone, 0), (right_zone, height), (255, 255, 255), 1)

                if target_found and target_cx is not None:
                    if target_cx < left_zone:
                        cv2.putText(frame, "MOTOR: SOL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif target_cx > right_zone:
                        cv2.putText(frame, "MOTOR: SAG", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "MOTOR: DURDU", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "MOTOR: DURDU (HEDEF YOK)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        self.root.after(15, self.update_frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    # Pencereleri sığdırmak için arayüzü tam ekran başlatalım
    root.state('zoomed') 
    app = YoloApp(root, "ESP32 Otonom Radar & Takip")
    root.mainloop()
