"""
ai_interface.py
---------------
Agentic AI Command Dispatcher for Air Drawing.
Enables Agentic AI, LLMs, or voice assistants to control the application using natural language commands:
  - "Change the color to blue"
  - "Clear the canvas"
  - "Save the drawing"
  - "Increase brush size"
  - "What did I draw?" / "Recognize shape" (Deep Learning StrokeCNN)
  - "Switch to Deep Learning" / "Switch to Machine Learning"
  - "Export analytics" / "Show performance stats"
"""

import re


class AICommandDispatcher:
    """
    Interprets natural language commands and executes them on the canvas and system.
    Provides an extensible interface ready for LLM Agents, Speech-to-Text,
    or automated script control.
    """

    def __init__(self, canvas_manager, app_reference=None):
        """
        :param canvas_manager: Instance of CanvasManager from drawing.py
        :param app_reference: Optional instance of AirDrawingApp from main.py
        """
        self.canvas = canvas_manager
        self.app = app_reference
        self.custom_handlers = {}

    def set_app(self, app_reference):
        """Set reference to the main app instance."""
        self.app = app_reference

    def execute_command(self, command_text):
        """
        Parse and dispatch a natural language command.
        :param command_text: e.g. 'Change the color to blue', 'What did I draw?', etc.
        :return: dict with 'status', 'action', and 'message'
        """
        if not command_text or not isinstance(command_text, str):
            return {"status": "error", "message": "Empty command"}

        cmd = command_text.strip().lower()

        # 1. Shape Recognition (Deep Learning)
        if any(w in cmd for w in ["recognize", "what did i draw", "identify", "detect shape"]):
            if self.app and hasattr(self.app, "recognize_current_drawing"):
                label, conf = self.app.recognize_current_drawing()
                return {
                    "status": "success",
                    "action": "recognize_shape",
                    "shape": label,
                    "confidence": conf,
                    "message": f"Deep Learning identified: {label} ({conf*100:.1f}%)"
                }
            return {"status": "error", "message": "App reference unavailable for shape recognition"}

        # 2. Analytics Commands
        elif any(w in cmd for w in ["analytics", "telemetry", "export stats", "report"]):
            if self.app and hasattr(self.app, "export_analytics"):
                path = self.app.export_analytics()
                return {"status": "success", "action": "export_analytics", "path": path, "message": "Analytics dashboard exported"}
            return {"status": "error", "message": "App reference unavailable for analytics export"}

        # 3. Model Engine Toggle
        elif "deep learning" in cmd or "dl mode" in cmd or "neural network" in cmd:
            if self.app and hasattr(self.app, "set_ai_engine"):
                self.app.set_ai_engine("DL")
                return {"status": "success", "action": "set_engine", "engine": "DL", "message": "Engine set to Deep Learning (DNN)"}
        elif "machine learning" in cmd or "ml mode" in cmd or "random forest" in cmd:
            if self.app and hasattr(self.app, "set_ai_engine"):
                self.app.set_ai_engine("ML")
                return {"status": "success", "action": "set_engine", "engine": "ML", "message": "Engine set to Machine Learning (Random Forest)"}
        elif "heuristic" in cmd or "rule" in cmd:
            if self.app and hasattr(self.app, "set_ai_engine"):
                self.app.set_ai_engine("HEURISTIC")
                return {"status": "success", "action": "set_engine", "engine": "HEURISTIC", "message": "Engine set to Heuristic Rules"}

        # 4. Color commands
        elif "blue" in cmd:
            self.canvas.set_color("BLUE")
            return {"status": "success", "action": "set_color", "color": "BLUE", "message": "Color changed to Blue"}
        elif "green" in cmd:
            self.canvas.set_color("GREEN")
            return {"status": "success", "action": "set_color", "color": "GREEN", "message": "Color changed to Green"}
        elif "red" in cmd:
            self.canvas.set_color("RED")
            return {"status": "success", "action": "set_color", "color": "RED", "message": "Color changed to Red"}
        elif "black" in cmd:
            self.canvas.set_color("BLACK")
            return {"status": "success", "action": "set_color", "color": "BLACK", "message": "Color changed to Black"}

        # 5. Clear canvas commands
        elif any(word in cmd for word in ["clear", "wipe", "clean", "reset"]):
            self.canvas.clear()
            return {"status": "success", "action": "clear", "message": "Canvas cleared"}

        # 6. Save drawing commands
        elif any(word in cmd for word in ["save", "export", "download"]):
            filepath = self.canvas.save()
            return {"status": "success", "action": "save", "filepath": filepath, "message": f"Drawing saved to {filepath}"}

        # 7. Brush size commands
        elif any(word in cmd for word in ["bigger", "increase", "large", "thick"]):
            self.canvas.set_brush_size("LARGE")
            return {"status": "success", "action": "brush_size", "size": "LARGE", "message": "Brush size set to Large"}
        elif any(word in cmd for word in ["smaller", "decrease", "small", "thin"]):
            self.canvas.set_brush_size("SMALL")
            return {"status": "success", "action": "brush_size", "size": "SMALL", "message": "Brush size set to Small"}
        elif "medium" in cmd:
            self.canvas.set_brush_size("MEDIUM")
            return {"status": "success", "action": "brush_size", "size": "MEDIUM", "message": "Brush size set to Medium"}

        # 8. Erase command
        elif "erase" in cmd:
            self.canvas.show_toast("Close fist to erase strokes")
            return {"status": "success", "action": "erase_hint", "message": "Eraser activated: Close fist over canvas to erase"}

        # 9. Custom handlers
        for intent, handler in self.custom_handlers.items():
            if intent in cmd:
                return handler(cmd)

        return {
            "status": "unrecognized",
            "message": f"Command '{command_text}' not recognized. Try 'blue', 'recognize shape', 'export analytics', or 'clear'."
        }

    def register_custom_action(self, keyword, handler_func):
        """Allow external Agentic AI modules to register custom actions."""
        self.custom_handlers[keyword.lower()] = handler_func
