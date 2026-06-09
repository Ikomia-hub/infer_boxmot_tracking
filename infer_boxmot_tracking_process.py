"""
Module that implements the core logic of algorithm execution.
"""
import copy
import os

import torch
import numpy as np

from ikomia import core, dataprocess
from ikomia.utils import strtobool

from infer_boxmot_tracking.core.tracker import BoxmotTracker
from infer_boxmot_tracking.core.utils import filter_objects, convert_objects_for_tracking, fill_tracks_output


class InferBoxmotTrackingParam(core.CWorkflowTaskParam):
    """
    Class to handle the algorithm parameters.
    Inherits PyCore.CWorkflowTaskParam from Ikomia API.
    """
    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.tracker = "occluboost"
        self.reid = "osnet_x0_25_msmt17"
        self.config = ""
        self.categories = "all"
        self.cuda = torch.cuda.is_available()
        self.half = torch.cuda.is_available()
        self.update = False

    def set_values(self, params):
        """
        Set parameters values from Ikomia Studio or API.
        Parameters values are stored as string and accessible like a python dict.
        """
        self.update = (
            strtobool(params["update"]) or
            self.cuda != strtobool(params["cuda"]) or
            self.half != strtobool(params["half"]) or
            self.tracker != params["tracker"] or
            self.reid != params["reid"] or
            self.config != params["config"]
        )
        self.cuda = strtobool(params["cuda"])
        self.half = strtobool(params["half"])
        self.tracker = params["tracker"]
        self.reid = params["reid"]
        self.config = params["config"]
        self.categories = params["categories"]

    def get_values(self):
        """
        Send parameters values to Ikomia Studio or API.
        Create the specific dict structure (key-value as string).
        """
        params = {
            "update": str(self.update),
            "cuda": str(self.cuda),
            "half": str(self.half),
            "tracker": self.tracker,
            "reid": self.reid,
            "config": self.config,
            "categories": self.categories,
        }
        return params


class InferBoxmotTrackingParamFactory(dataprocess.CTaskParamFactory):
    """Factory class to create parameters object."""
    def __init__(self):
        dataprocess.CTaskParamFactory.__init__(self)
        self.name = "infer_boxmot_tracking"

    def create(self):
        """Instantiate parameters object."""
        return InferBoxmotTrackingParam()


class InferBoxmotTracking(dataprocess.C2dImageTask):
    """
    Class that implements the algorithm.
    Inherits PyCore.CWorkflowTask or derived from Ikomia API.
    """
    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.remove_input(1)
        self.add_input(dataprocess.CObjectDetectionIO())
        self.add_input(dataprocess.CInstanceSegmentationIO())
        self.add_input(dataprocess.CKeypointsIO())

        self.add_output(dataprocess.CInstanceSegmentationIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferBoxmotTrackingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.tracker = None
        self.model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    def get_progress_steps(self):
        """
        Ikomia Studio only.
        Function returning the number of progress steps for this algorithm.
        This is handled by the main progress bar of Ikomia Studio.
        """
        return 1

    def _get_input_to_track(self):
        inputs = [self.get_input(1), self.get_input(2), self.get_input(3)]
        for candidate_input in inputs:
            if candidate_input.is_data_available():
                return candidate_input

        return None

    def _update_output_type(self, input_img, input_to_track):
        input_type = type(input_to_track)
        if not isinstance(self.get_output(1), input_type):
            self.remove_output(1)
            self.add_output(input_type())

            output = self.get_output(1)
            if isinstance(output, dataprocess.CInstanceSegmentationIO):
                h, w = np.shape(input_img)[:2]
                output.init("Boxmot tracking", 0, w, h)
            else:
                output.init("Boxmot tracking", 0)

    def run(self):
        """Main function and entry point for algorithm execution."""
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters:
        param = self.get_param_object()

        if self.tracker is None or param.update:
            device = torch.device('cuda' if param.cuda and torch.cuda.is_available() else 'cpu')
            half = param.half and param.cuda and torch.cuda.is_available()

            self.tracker = BoxmotTracker(
                tracker_name=param.tracker,
                reid_weight=os.path.join(self.model_folder, param.reid + ".pt"),
                config_path=param.config or None,
                device=device,
                half=half
            )
            param.update = False

        input_to_track = self._get_input_to_track()
        if input_to_track is None:
            self.end()
            return

        img_in = self.get_input(0).get_image()
        self._update_output_type(img_in, input_to_track)

        objs_to_track = filter_objects(input_to_track, param.categories)
        if not objs_to_track:
            self.end()
            return

        track_input = convert_objects_for_tracking(objs_to_track)
        tracks = self.tracker.update(track_input, img_in)
        fill_tracks_output(type(input_to_track), objs_to_track, self.get_output(1), tracks)
        self.end()

    def end(self):
        self.forward_input_image(0, 0)
        self.emit_step_progress()
        self.end_task_run()


class InferBoxmotTrackingFactory(dataprocess.CTaskFactory):
    """
    Factory class to create process object.
    Inherits PyDataProcess.CTaskFactory from Ikomia API.
    """
    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_boxmot_tracking"
        self.info.short_description = "Multiple object tracking algorithm for object detection, instance segmentation and keypoints"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Tracking"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = ""
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2026
        self.info.license = "AGPL-3.0 license"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.16.1"
        # self.info.max_ikomia_version = "0.16.1"

        # Python compatibility
        # self.info.min_python_version = "3.10.0"
        # self.info.max_python_version = "3.13.0"

        # URL of documentation
        self.info.documentation_link = ""

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_boxmot_tracking"
        self.info.original_repository = "https://github.com/mikel-brostrom"

        # Keywords used for search
        self.info.keywords = "object,tracking,detector"

        # General type: INFER, TRAIN, DATASET or OTHER
        self.info.algo_type = core.AlgoType.INFER

        # Algorithms tasks: CLASSIFICATION, COLORIZATION, IMAGE_CAPTIONING, IMAGE_GENERATION,
        # IMAGE_MATTING, INPAINTING, INSTANCE_SEGMENTATION, KEYPOINTS_DETECTION,
        # OBJECT_DETECTION, OBJECT_TRACKING, OCR, OPTICAL_FLOW, OTHER, PANOPTIC_SEGMENTATION,
        # SEMANTIC_SEGMENTATION or SUPER_RESOLUTION
        self.info.algo_tasks = "OBJECT_TRACKING"

        # Min hardware config
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 4

    def create(self, param=None):
        """Instantiate algorithm object."""
        return InferBoxmotTracking(self.info.name, param)
