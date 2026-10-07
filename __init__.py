from .nodes import FrameAuditAnalyze

__version__ = "0.1.0"
NODE_CLASS_MAPPINGS = {"FrameAuditAnalyze": FrameAuditAnalyze}
NODE_DISPLAY_NAME_MAPPINGS = {"FrameAuditAnalyze": "FrameAudit · Analyze Clip"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
