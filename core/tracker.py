import torch

from boxmot.trackers.tracker_zoo import create_tracker


class BoxmotTracker:
    def __init__(
            self,
            tracker_name: str,
            reid_weight: str,
            config_path: str,
            device: torch.device,
            half: bool):
        self.tracker = create_tracker(
            tracker_type=tracker_name,
            tracker_config=config_path,
            reid_weights=reid_weight,
            device=device,
            half=half
        )

    def update(self, detections, input_image):
        return self.tracker.update(detections, input_image)
