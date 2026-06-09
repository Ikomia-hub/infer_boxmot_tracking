"""
Module that implements the UI widget of the algorithm.
"""
from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion

# PyQt GUI framework
from PyQt6.QtWidgets import *

from infer_boxmot_tracking.infer_boxmot_tracking_process import InferBoxmotTrackingParam
from infer_boxmot_tracking.core.utils import TRACKERS, TRACKER_USE_REID, get_reid_models


class InferBoxmotTrackingWidget(core.CWorkflowTaskWidget):
    """
    Class that implements UI widget to adjust algorithm parameters.
    Inherits PyCore.CWorkflowTaskWidget from Ikomia API.
    """
    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferBoxmotTrackingParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Fill layout
        # Trackers
        self.combo_tracker = pyqtutils.append_combo(self.grid_layout, "Trackers")
        for tracker in TRACKERS:
            self.combo_tracker.addItem(tracker)

        self.combo_tracker.setCurrentText(self.parameters.tracker)
        self.combo_tracker.currentIndexChanged.connect(self.on_tracker_changed)

        # REIDs
        self.combo_reid = pyqtutils.append_combo(self.grid_layout, "REID models")
        for reid in get_reid_models():
            self.combo_reid.addItem(reid)

        self.combo_reid.setCurrentText(self.parameters.reid)

        # Config
        self.browse_config = pyqtutils.append_browse_file(
            self.grid_layout,
            label="Tracker config",
            path=self.parameters.config,
            file_filter="*.yaml",
        )

        self.edit_categories = pyqtutils.append_edit(self.grid_layout, "Categories", self.parameters.categories)
        self.check_cuda = pyqtutils.append_check(self.grid_layout, "Cuda", self.parameters.cuda)
        self.check_half = pyqtutils.append_check(self.grid_layout, "Half precision", self.parameters.half)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        # Set widget layout
        self.set_layout(layout_ptr)

    def on_tracker_changed(self, index):
        tracker_name = self.combo_tracker.currentText()
        if tracker_name in TRACKER_USE_REID:
            self.combo_reid.setEnabled(TRACKER_USE_REID[tracker_name])
        else:
            self.combo_reid.setEnabled(False)

    def on_apply(self):
        """QT slot called when users click the Apply button."""
        self.parameters.update = (
            self.parameters.cuda != self.check_cuda.isChecked() or
            self.parameters.half != self.check_half.isChecked() or
            self.parameters.tracker != self.combo_tracker.currentText() or
            self.parameters.reid != self.combo_reid.currentText() or
            (self.browse_config.path and self.parameters.config != self.browse_config.path)
        )

        # Get parameters from widget
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.half = self.check_half.isChecked()
        self.parameters.tracker = self.combo_tracker.currentText()
        self.parameters.reid = self.combo_reid.currentText()
        self.parameters.categories = self.edit_categories.text()
        self.parameters.config = self.browse_config.path

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


class InferBoxmotTrackingWidgetFactory(dataprocess.CWidgetFactory):
    """
    Factory class to create algorithm widget object.
    Inherits PyDataProcess.CWidgetFactory from Ikomia API.
    """
    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_boxmot_tracking"

    def create(self, param):
        """Instantiate widget object."""
        return InferBoxmotTrackingWidget(param, None)
