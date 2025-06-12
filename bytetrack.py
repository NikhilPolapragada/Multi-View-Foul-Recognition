import cv2
import numpy as np
from collections import deque

class SimpleTracker:
    def __init__(self, max_lost=30):
        self.next_object_id = 0
        self.objects = {}  # object_id -> centroid
        self.lost = {}     # object_id -> number of consecutive lost frames
        self.max_lost = max_lost

    def update(self, detections):
        objects_bbs_ids = []
        centroids = np.array([(int((x1+x2)/2), int((y1+y2)/2)) for x1, y1, x2, y2 in detections])
        
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.objects[self.next_object_id] = centroids[i]
                self.lost[self.next_object_id] = 0
                objects_bbs_ids.append((detections[i], self.next_object_id))
                self.next_object_id += 1
        else:
            new_objects = {}
            new_lost = {}

            for object_id in self.objects:
                self.lost[object_id] += 1
                if self.lost[object_id] > self.max_lost:
                    continue
                new_objects[object_id] = self.objects[object_id]
                new_lost[object_id] = self.lost[object_id]
            
            self.objects = new_objects
            self.lost = new_lost

            for i in range(len(detections)):
                self.objects[self.next_object_id] = centroids[i]
                self.lost[self.next_object_id] = 0
                objects_bbs_ids.append((detections[i], self.next_object_id))
                self.next_object_id += 1
        
        return objects_bbs_ids

# Example usage
def run_tracker_on_video(video_path, detections_per_frame):
    tracker = SimpleTracker()
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detections_per_frame.get(frame_id, [])
        tracked = tracker.update(detections)

        for (x1, y1, x2, y2), object_id in tracked:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, str(object_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        frame_id += 1

    cap.release()

