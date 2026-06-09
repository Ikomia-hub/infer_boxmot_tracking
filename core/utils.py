from pathlib import Path

import numpy as np

from boxmot.reid.core.config import TRAINED_URLS

from ikomia.dataprocess import CObjectDetectionIO, CInstanceSegmentationIO, CKeypointsIO


TRACKERS = [
    "occluboost",
    "botsort",
    "boosttrack",
    "strongsort",
    "deepocsort",
    "bytetrack",
    "hybridsort",
    "ocsort",
    "sfsort",
]


TRACKER_USE_REID = {
    "occluboost": True,
    "botsort": True,
    "boosttrack": True,
    "strongsort": True,
    "deepocsort": True,
    "bytetrack": False,
    "hybridsort": True,
    "ocsort": False,
    "sfsort": False,
}

PALETTE = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def get_reid_models() -> list:
    return [str(Path(weight_name).with_suffix("")) for weight_name in TRAINED_URLS]

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def filter_objects(input_to_track, categories):
    objs = input_to_track.get_objects()
    if categories == "all":
        return objs

    labels_to_track = {part.strip() for part in categories.split(",")}
    return [det for det in objs if det.label in labels_to_track]


def convert_objects_for_tracking(objs_to_track):
    label_to_id = {}
    dets = []

    for det in objs_to_track:
        if det.label not in label_to_id:
            label_to_id[det.label] = len(label_to_id)

        # --> (x, y, x, y, id, conf, cls, ind)
        dets.append([*xywh_to_xyxy(det.box), det.confidence, label_to_id[det.label]])

    return np.array(dets)


def compute_color_for_track(track_id):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (track_id ** 2 - track_id + 1)) % 255) for p in PALETTE]
    return color


def fill_tracks_output(input_type, objs_to_track, ik_output, tracks):
    ids = tracks[:, 4].astype('int')  # float64 to int
    indices = tracks[:, 7].astype('int')  # float64 to int

    for index, track_id in zip(indices, ids):
        track_id = int(track_id)
        obj = objs_to_track[index]
        color = compute_color_for_track(track_id)

        if input_type == CObjectDetectionIO:
            obj_args = (track_id, obj.label, obj.confidence, *obj.box, color)
        elif input_type == CInstanceSegmentationIO:
            obj_args = (track_id, obj.type, obj.class_index, obj.label, obj.confidence, *obj.box, obj.mask, color)
        elif input_type == CKeypointsIO:
            obj_args = (track_id, obj.label, obj.confidence, *obj.box, obj.points, color)
        else:
            raise RuntimeError("Invalid input type, valid types are CObjectDetectionIO, CInstanceSegmentationIO or CKeypointsIO")

        ik_output.add_object(*obj_args)
