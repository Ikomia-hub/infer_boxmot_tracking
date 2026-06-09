"""
Main Ikomia plugin module.
Ikomia Studio and Ikomia API use it to load algorithms dynamically.
"""
from ikomia import dataprocess
from infer_boxmot_tracking.infer_boxmot_tracking_process import InferBoxmotTrackingFactory
from infer_boxmot_tracking.infer_boxmot_tracking_process import InferBoxmotTrackingParamFactory


class IkomiaPlugin(dataprocess.CPluginProcessInterface):
    """
    Interface class to integrate the process with Ikomia application.
    Inherits PyDataProcess.CPluginProcessInterface from Ikomia API.
    """
    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        """Instantiate process object."""
        return InferBoxmotTrackingFactory()

    def get_widget_factory(self):
        """Instantiate associated widget object."""
        from infer_boxmot_tracking.infer_boxmot_tracking_widget import InferBoxmotTrackingWidgetFactory
        return InferBoxmotTrackingWidgetFactory()

    def get_param_factory(self):
        """Instantiate algorithm parameters object."""
        return InferBoxmotTrackingParamFactory()
