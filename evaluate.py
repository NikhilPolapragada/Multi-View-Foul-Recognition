import os
import torch
from models.detr import DETRModel
from models.bytetrack import SimpleTracker
from models.timesformer import TimeSformerClassifier
from utils.dataset import load_test_videos

FOUL_CLASSES = ['No Offence', 'Offence + No Card', 'Offence + Yellow Card', 'Offence + Red Card']
ACTION_CLASSES = ['Standing Tackle', 'Tackle', 'Holding', 'Pushing', 'Challenge', 'Dive', 'High Leg', 'Elbowing']

def evaluate(test_dir, detr_path, timesformer_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    detr = DETRModel().to(device)
    detr.load_state_dict(torch.load(detr_path, map_location=device))
    detr.eval()

    timesformer = TimeSformerClassifier().to(device)
    timesformer.load_state_dict(torch.load(timesformer_path, map_location=device))
    timesformer.eval()

    tracker = SimpleTracker()

    test_videos = load_test_videos(test_dir)

    results = []

    for video_path in test_videos:
        print(f"Processing {video_path}...")
        video_tensor = torch.load(video_path).to(device)

        with torch.no_grad():
            # Run detection per frame
            detections = []
            for frame in video_tensor:
                pred_boxes = detr(frame.unsqueeze(0))  # [1, 3, 224, 224]
                detections.append(pred_boxes)

            # Tracking (simplified)
            detections_per_frame = {i: det for i, det in enumerate(detections)}
            tracker.update(detections)

            # Classification (entire clip)
            foul_pred, action_pred = timesformer(video_tensor.unsqueeze(0))
            foul_top2 = torch.topk(foul_pred, 2).indices.squeeze().cpu().tolist()
            action_top2 = torch.topk(action_pred, 2).indices.squeeze().cpu().tolist()

            result = {
                "video": os.path.basename(video_path),
                "foul_preds": [FOUL_CLASSES[i] for i in foul_top2],
                "action_preds": [ACTION_CLASSES[i] for i in action_top2]
            }
            results.append(result)
            print(result)

    return results

if __name__ == "__main__":
    test_dir = "data/test"
    detr_weights = "weights/detr.pth"
    timesformer_weights = "weights/timesformer.pth"
    
    results = evaluate(test_dir, detr_weights, timesformer_weights)

    # Save or evaluate results
    with open("test/results.txt", "w") as f:
        for r in results:
            f.write(str(r) + "\n")

