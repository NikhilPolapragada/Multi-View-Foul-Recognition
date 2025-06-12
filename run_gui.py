import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import cv2
from models.detr import DETRModel
from models.bytetrack import SimpleTracker
from models.timesformer import TimeSformerClassifier
from utils.preprocessing import preprocess_video

FOUL_CLASSES = ['No Offence', 'Offence + No Card', 'Offence + Yellow Card', 'Offence + Red Card']
ACTION_CLASSES = ['Standing Tackle', 'Tackle', 'Holding', 'Pushing', 'Challenge', 'Dive', 'High Leg', 'Elbowing']

class FoulRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Foul Recognition")

        self.label = tk.Label(root, text="Select a video to analyze", font=("Arial", 14))
        self.label.pack(pady=10)

        self.button = tk.Button(root, text="Browse Video", command=self.load_video)
        self.button.pack(pady=5)

        self.result_text = tk.Text(root, height=10, width=60, font=("Arial", 12))
        self.result_text.pack(pady=10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models()

    def load_models(self):
        self.detr = DETRModel().to(self.device)
        self.detr.load_state_dict(torch.load("weights/detr.pth", map_location=self.device))
        self.detr.eval()

        self.tracker = SimpleTracker()

        self.timesformer = TimeSformerClassifier().to(self.device)
        self.timesformer.load_state_dict(torch.load("weights/timesformer.pth", map_location=self.device))
        self.timesformer.eval()

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not file_path:
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Processing {file_path}...\n")

        try:
            frames = preprocess_video(file_path)  # [T, 3, H, W]
            frames = frames.to(self.device)

            # DETR detections
            detections = []
            with torch.no_grad():
                for frame in frames:
                    pred = self.detr(frame.unsqueeze(0))
                    detections.append(pred)
                self.tracker.update(detections)

                # TimeSformer classification
                foul_pred, action_pred = self.timesformer(frames.unsqueeze(0))
                foul_top2 = torch.topk(foul_pred, 2).indices.squeeze().cpu().tolist()
                action_top2 = torch.topk(action_pred, 2).indices.squeeze().cpu().tolist()

            self.result_text.insert(tk.END, "\nTop-2 Foul Predictions:\n")
            for i in foul_top2:
                self.result_text.insert(tk.END, f"- {FOUL_CLASSES[i]}\n")

            self.result_text.insert(tk.END, "\nTop-2 Action Type Predictions:\n")
            for i in action_top2:
                self.result_text.insert(tk.END, f"- {ACTION_CLASSES[i]}\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = FoulRecognitionApp(root)
    root.mainloop()

