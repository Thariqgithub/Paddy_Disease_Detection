import tkinter as tk
from tkinter import ttk, font
import cv2
import threading
import time
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np


DISEASE_INFO = {
    "Bacterial_Blight": {
        "label": "Bacterial Blight",
        "icon": "🦠",
        "color": "#FF6B6B",
        "tag_color": "#FF4444",
        "reason": (
            "Caused by the bacterium Xanthomonas oryzae pv. oryzae. "
            "It spreads through infected water, contaminated tools, and wind-driven rain. "
            "The bacteria enter through leaf margins and water pores, causing water-soaked "
            "lesions that turn yellowish-white and spread along leaf edges."
        ),
        "solutions": [
            "Remove and destroy infected plant material immediately to prevent spread.",
            "Apply copper-based bactericides (e.g., copper hydroxide) as a foliar spray.",
            "Avoid overhead irrigation; use drip irrigation to keep foliage dry.",
            "Use certified disease-free seeds and resistant varieties (e.g., IR64, Swarna).",
            "Maintain balanced fertilization — avoid excessive nitrogen which promotes spread.",
            "Drain fields periodically and improve drainage to reduce waterlogging.",
            "Rotate crops with non-host plants in subsequent seasons.",
        ],
    },
    "Brown_Spot": {
        "label": "Brown Spot",
        "icon": "🟤",
        "color": "#D4A057",
        "tag_color": "#B8842A",
        "reason": (
            "Caused by the fungus Cochliobolus miyabeanus (formerly Helminthosporium oryzae). "
            "It thrives under nutrient-deficient (especially potassium & silicon) conditions, "
            "high humidity, and temperatures between 25–35 °C. Spores spread via wind and rain splash."
        ),
        "solutions": [
            "Apply potassium and silicon fertilizers to strengthen cell walls and improve resistance.",
            "Treat seeds with fungicides (Thiram or Carbendazim) before sowing.",
            "Spray Propiconazole (Tilt 25 EC) or Mancozeb at first sign of infection.",
            "Maintain optimal plant nutrition — avoid deficiencies in N, K, and Zn.",
            "Use resistant varieties wherever available.",
            "Remove crop debris after harvest to eliminate fungal overwintering sites.",
            "Ensure proper plant spacing for adequate air circulation.",
        ],
    },
    "Healthy-Plant": {
        "label": "Healthy Plant",
        "icon": "🌿",
        "color": "#4CAF50",
        "tag_color": "#2E7D32",
        "reason": (
            "No disease detected. The plant appears healthy with normal green coloration, "
            "no visible lesions or spots. Continue current agronomic practices and routine monitoring."
        ),
        "solutions": [
            "Maintain regular irrigation schedule appropriate for the growth stage.",
            "Continue balanced NPK fertilization program.",
            "Monitor weekly for early signs of pests or disease.",
            "Ensure proper field drainage to prevent waterlogging.",
            "Keep weeds controlled to reduce competition and pest habitat.",
            "Record observations for future crop planning.",
        ],
    },
    "Rice_Blast": {
        "label": "Rice Blast",
        "icon": "💥",
        "color": "#E57373",
        "tag_color": "#C62828",
        "reason": (
            "Caused by the fungus Magnaporthe oryzae — one of the most devastating rice diseases globally. "
            "It spreads rapidly via airborne conidia under cool nights (below 20 °C), humid conditions, "
            "and heavy dew. Lesions appear as diamond-shaped grey spots with brown borders on leaves, "
            "nodes, and panicles."
        ),
        "solutions": [
            "Apply Tricyclazole or Isoprothiolane fungicides at the tillering and heading stages.",
            "Avoid excessive nitrogen fertilization which increases susceptibility.",
            "Plant blast-resistant varieties (e.g., IR72, MTU7029, Swarna Sub1).",
            "Ensure adequate potassium levels to improve plant resilience.",
            "Avoid late transplanting which increases exposure during vulnerable growth stages.",
            "Use silicon-based fertilizers to strengthen leaf tissue.",
            "Drain fields for 3–5 days during peak infection risk periods.",
            "Destroy infected stubble and ratoons after harvest.",
        ],
    },
}

NO_DETECTION = {
    "label": "No Detection",
    "icon": "🔍",
    "color": "#78909C",
    "tag_color": "#546E7A",
    "reason": "Point the camera at rice leaves to begin detection.",
    "solutions": ["Ensure the camera has a clear view of the rice plant foliage."],
}


BG_DARK    = "#0D1117"
BG_CARD    = "#161B22"
BG_CARD2   = "#1C2128"
ACCENT     = "#58A6FF"
TEXT_PRI   = "#E6EDF3"
TEXT_SEC   = "#8B949E"
BORDER     = "#30363D"
GREEN_LIVE = "#3FB950"


# ─────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────
class RiceDiseaseDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Rice Leaf Disease Detection  •  YOLOv8")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1200x820")
        self.root.minsize(1000, 700)

        # State
        self.model = None
        self.cap = None
        self.running = False
        self.current_class = None
        self.current_conf = 0.0
        self._frame_rgb = None
        self._lock = threading.Lock()

        self._build_ui()
        # Schedule model loading & camera after main loop starts
        self.root.after(100, self._load_model)
        self.root.after(100, self._start_camera)

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=BG_DARK, pady=10)
        header.pack(fill="x", padx=20)

        title_f = tk.Frame(header, bg=BG_DARK)
        title_f.pack(side="left")
        tk.Label(title_f, text="🌾", font=("Segoe UI Emoji", 22), bg=BG_DARK).pack(side="left", padx=(0, 8))
        tk.Label(title_f, text="Rice Leaf Disease Detector",
                 font=("Segoe UI", 16, "bold"), bg=BG_DARK, fg=TEXT_PRI).pack(side="left")
        tk.Label(title_f, text="  YOLOv8 Real-Time",
                 font=("Segoe UI", 10), bg=BG_DARK, fg=TEXT_SEC).pack(side="left")

        # Live indicator
        self.live_frame = tk.Frame(header, bg=BG_DARK)
        self.live_frame.pack(side="right")
        self.live_dot = tk.Label(self.live_frame, text="●", font=("Segoe UI", 12),
                                  bg=BG_DARK, fg=GREEN_LIVE)
        self.live_dot.pack(side="left")
        tk.Label(self.live_frame, text=" LIVE", font=("Segoe UI", 10, "bold"),
                 bg=BG_DARK, fg=GREEN_LIVE).pack(side="left")

        # ── Separator ──
        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill="x", padx=20)

        # ── Main Content ──
        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill="both", expand=True, padx=20, pady=12)

        # Left column — camera + detection badge
        left_col = tk.Frame(content, bg=BG_DARK)
        left_col.pack(side="left", fill="y")
        self._build_camera_panel(left_col)

        # Right column — info panels
        right_col = tk.Frame(content, bg=BG_DARK)
        right_col.pack(side="left", fill="both", expand=True, padx=(16, 0))
        self._build_info_panel(right_col)

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Initializing…")
        status_bar = tk.Frame(self.root, bg=BG_CARD2, pady=5)
        status_bar.pack(fill="x", side="bottom")
        tk.Label(status_bar, textvariable=self.status_var,
                 font=("Segoe UI", 9), bg=BG_CARD2, fg=TEXT_SEC, padx=16).pack(side="left")

    def _build_camera_panel(self, parent):
        # Camera frame
        cam_card = tk.Frame(parent, bg=BG_CARD, bd=0, relief="flat",
                            highlightthickness=1, highlightbackground=BORDER)
        cam_card.pack()

        self.cam_label = tk.Label(cam_card, bg="#000000", width=640, height=480)
        self.cam_label.pack(padx=2, pady=2)

        # Detection badge
        badge_frame = tk.Frame(parent, bg=BG_DARK, pady=8)
        badge_frame.pack(fill="x")

        self.badge_bg = tk.Frame(badge_frame, bg=BG_CARD, bd=0,
                                  highlightthickness=1, highlightbackground=BORDER)
        self.badge_bg.pack(fill="x")

        inner = tk.Frame(self.badge_bg, bg=BG_CARD, padx=16, pady=12)
        inner.pack(fill="x")

        left_b = tk.Frame(inner, bg=BG_CARD)
        left_b.pack(side="left", fill="y")
        self.badge_icon = tk.Label(left_b, text="🔍", font=("Segoe UI Emoji", 26),
                                    bg=BG_CARD, fg=TEXT_PRI)
        self.badge_icon.pack()

        right_b = tk.Frame(inner, bg=BG_CARD, padx=12)
        right_b.pack(side="left", fill="both", expand=True)

        self.badge_class = tk.Label(right_b, text="No Detection",
                                     font=("Segoe UI", 13, "bold"), bg=BG_CARD, fg=TEXT_PRI)
        self.badge_class.pack(anchor="w")

        self.badge_conf = tk.Label(right_b, text="Confidence: —",
                                    font=("Segoe UI", 9), bg=BG_CARD, fg=TEXT_SEC)
        self.badge_conf.pack(anchor="w")

        # Confidence bar
        bar_container = tk.Frame(right_b, bg=BG_CARD, pady=4)
        bar_container.pack(fill="x", anchor="w")
        bar_bg = tk.Frame(bar_container, bg=BORDER, height=6, bd=0)
        bar_bg.pack(fill="x")
        bar_bg.pack_propagate(False)
        self.conf_bar = tk.Frame(bar_bg, bg=ACCENT, height=6)
        self.conf_bar.place(x=0, y=0, relheight=1.0, relwidth=0.0)

    def _build_info_panel(self, parent):
        # ── Reason card ──
        reason_lbl = tk.Label(parent, text="DISEASE CAUSE",
                               font=("Segoe UI", 8, "bold"), bg=BG_DARK,
                               fg=TEXT_SEC, anchor="w")
        reason_lbl.pack(fill="x", pady=(0, 4))

        reason_card = tk.Frame(parent, bg=BG_CARD, bd=0,
                                highlightthickness=1, highlightbackground=BORDER)
        reason_card.pack(fill="x", pady=(0, 14))

        reason_inner = tk.Frame(reason_card, bg=BG_CARD, padx=16, pady=12)
        reason_inner.pack(fill="both")

        self.reason_text = tk.Text(reason_inner, bg=BG_CARD, fg=TEXT_PRI,
                                    font=("Segoe UI", 10), wrap="word",
                                    relief="flat", bd=0, height=5,
                                    state="disabled", cursor="arrow",
                                    highlightthickness=0)
        self.reason_text.pack(fill="both", expand=True)

        # ── Solutions card ──
        sol_lbl = tk.Label(parent, text="TREATMENT & SOLUTIONS",
                            font=("Segoe UI", 8, "bold"), bg=BG_DARK,
                            fg=TEXT_SEC, anchor="w")
        sol_lbl.pack(fill="x", pady=(0, 4))

        sol_card = tk.Frame(parent, bg=BG_CARD, bd=0,
                             highlightthickness=1, highlightbackground=BORDER)
        sol_card.pack(fill="both", expand=True)

        # Scrollable solutions
        sol_scroll_frame = tk.Frame(sol_card, bg=BG_CARD)
        sol_scroll_frame.pack(fill="both", expand=True, padx=2, pady=2)

        canvas = tk.Canvas(sol_scroll_frame, bg=BG_CARD, bd=0,
                            highlightthickness=0)
        scrollbar = ttk.Scrollbar(sol_scroll_frame, orient="vertical",
                                   command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.sol_inner = tk.Frame(canvas, bg=BG_CARD)
        self.sol_window = canvas.create_window((0, 0), window=self.sol_inner, anchor="nw")

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_configure(e):
            canvas.itemconfig(self.sol_window, width=e.width)

        self.sol_inner.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self.solutions_canvas = canvas
        self._render_disease_info(NO_DETECTION)

    # ── Logic ────────────────────────────────────────────────────────────────

    def _load_model(self):
        def _load():
            try:
                self.root.after(0, lambda: self.status_var.set("Loading YOLOv8 model  best.pt …"))
                self.model = YOLO("best.pt")
                self.root.after(0, lambda: self.status_var.set("Model loaded successfully  ✓"))

            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Model error: {e}"))
        threading.Thread(target=_load, daemon=True).start()

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("⚠ Camera not found. Check connection.")
            return
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        self._update_frame()

    def _capture_loop(self):
        """Background thread: capture + run inference."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            best_class = None
            best_conf  = 0.0

            if self.model is not None:
                try:
                    results = self.model(frame, verbose=False, conf=0.35)
                    for r in results:
                        if r.boxes is not None and len(r.boxes):
                            for box in r.boxes:
                                conf  = float(box.conf[0])
                                cls   = int(box.cls[0])
                                label = self.model.names[cls]
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                info  = DISEASE_INFO.get(label, {})
                                color_hex = info.get("color", "#58A6FF")
                                bgr = self._hex_to_bgr(color_hex)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                                text = f"{label.replace('_',' ')}  {conf:.0%}"
                                (tw, th), _ = cv2.getTextSize(
                                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                                cv2.rectangle(frame, (x1, y1-th-10),
                                              (x1+tw+8, y1), bgr, -1)
                                cv2.putText(frame, text, (x1+4, y1-4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                            (15, 17, 23), 1, cv2.LINE_AA)
                                if conf > best_conf:
                                    best_conf  = conf
                                    best_class = label
                except Exception as e:
                    self.root.after(0, lambda err=e: self.status_var.set(f"Inference error: {err}"))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame_rgb  = rgb
                self.current_class = best_class
                self.current_conf  = best_conf

    def _update_frame(self):
        """Main-thread loop: render latest frame + update UI."""
        with self._lock:
            frame_rgb  = self._frame_rgb
            det_class  = self.current_class
            det_conf   = self.current_conf

        if frame_rgb is not None:
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=photo)
            self.cam_label.image = photo

            # Update badge + info panels when detection changes
            if det_class != self._last_class:
                self._last_class = det_class
                info = DISEASE_INFO.get(det_class, NO_DETECTION) if det_class else NO_DETECTION
                self._update_badge(info, det_conf)
                self._render_disease_info(info)

            elif det_class:
                # Just update confidence bar & label each frame
                self.badge_conf.configure(
                    text=f"Confidence: {det_conf:.1%}")
                bar_w = max(0.0, min(1.0, det_conf))
                self.conf_bar.place(relwidth=bar_w)
                self.status_var.set(
                    f"Detecting  ▸  {det_class.replace('_',' ')}  ({det_conf:.1%})")

        if self.running:
            self.root.after(30, self._update_frame)

    _last_class = "__init__"   # sentinel

    def _update_badge(self, info: dict, conf: float):
        color = info["color"]
        tag   = info["tag_color"]
        self.badge_icon.configure(text=info["icon"])
        self.badge_class.configure(text=info["label"], fg=color)
        self.badge_conf.configure(
            text=f"Confidence: {conf:.1%}" if conf else "Confidence: —")
        self.conf_bar.configure(bg=color)
        bar_w = max(0.0, min(1.0, conf))
        self.conf_bar.place(relwidth=bar_w)
        # Accent border on badge
        self.badge_bg.configure(highlightbackground=tag)

    def _render_disease_info(self, info: dict):
        # Reason
        self.reason_text.configure(state="normal")
        self.reason_text.delete("1.0", "end")
        self.reason_text.insert("end", info["reason"])
        self.reason_text.configure(state="disabled")

        # Solutions — clear existing widgets
        for w in self.sol_inner.winfo_children():
            w.destroy()

        color = info.get("color", ACCENT)
        for i, step in enumerate(info["solutions"], 1):
            row = tk.Frame(self.sol_inner, bg=BG_CARD)
            row.pack(fill="x", padx=16, pady=4)

            num_bg = tk.Frame(row, bg=color, width=26, height=26)
            num_bg.pack(side="left", anchor="n")
            num_bg.pack_propagate(False)
            tk.Label(num_bg, text=str(i), font=("Segoe UI", 9, "bold"),
                     bg=color, fg=BG_DARK).pack(expand=True)

            tk.Label(row, text=step, font=("Segoe UI", 10),
                     bg=BG_CARD, fg=TEXT_PRI, wraplength=380,
                     justify="left", anchor="nw", padx=10).pack(
                side="left", fill="x", expand=True)

        self.solutions_canvas.yview_moveto(0)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _hex_to_bgr(hex_color: str):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def on_close(self):
        self.running = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = RiceDiseaseDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Style scrollbars
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Vertical.TScrollbar",
                    background=BG_CARD2, troughcolor=BG_CARD,
                    arrowcolor=TEXT_SEC, bordercolor=BORDER,
                    lightcolor=BG_CARD2, darkcolor=BG_CARD2)

    root.mainloop()