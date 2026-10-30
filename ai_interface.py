"""
ai_interface.py
---------------
Simple, beginner-friendly AI-ready Command Dispatcher for Air Drawing.
Enables Agentic AI or voice assistants to control the application using natural language commands:
  - "Change the color to blue"
  - "Clear the canvas"
  - "Erase"
  - "Save the drawing"
  - "Increase brush size"
"""

import re


class AICommandDispatcher:
    """
    Interprets natural language commands and executes them on the canvas.
    This provides an extensible interface ready for LLM Agents, Speech-to-Text,
    or automated script control.
    """

    def __init__(self, canvas_manager):
        """
        :param canvas_manager: Instance of CanvasManager from drawing.py
        """
        self.canvas = canvas_manager
        self.custom_handlers = {}

    def execute_command(self, command_text):
        """
        Parse and dispatch a natural language command.
        :param command_text: e.g. 'Change the color to blue', 'Clear the canvas', etc.
        :return: dict with 'status', 'action', and 'message'
        """
        if not command_text or not isinstance(command_text, str):
            return {"status": "error", "message": "Empty command"}

        cmd = command_text.strip().lower()

        # 1. Color commands
        if "blue" in cmd:
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

        # 2. Clear canvas commands
        elif any(word in cmd for word in ["clear", "wipe", "clean", "reset"]):
            self.canvas.clear()
            return {"status": "success", "action": "clear", "message": "Canvas cleared"}

        # 3. Save drawing commands
        elif any(word in cmd for word in ["save", "export", "download"]):
            filepath = self.canvas.save()
            return {"status": "success", "action": "save", "filepath": filepath, "message": f"Drawing saved to {filepath}"}

        # 4. Brush size commands
        elif any(word in cmd for word in ["bigger", "increase", "large", "thick"]):
            self.canvas.set_brush_size("LARGE")
            return {"status": "success", "action": "brush_size", "size": "LARGE", "message": "Brush size set to Large"}
        elif any(word in cmd for word in ["smaller", "decrease", "small", "thin"]):
            self.canvas.set_brush_size("SMALL")
            return {"status": "success", "action": "brush_size", "size": "SMALL", "message": "Brush size set to Small"}
        elif "medium" in cmd:
            self.canvas.set_brush_size("MEDIUM")
            return {"status": "success", "action": "brush_size", "size": "MEDIUM", "message": "Brush size set to Medium"}

        # 5. Erase command
        elif "erase" in cmd:
            self.canvas.show_toast("Close fist to erase strokes")
            return {"status": "success", "action": "erase_hint", "message": "Eraser activated: Close fist over canvas to erase"}

        # 6. Check custom registered handlers (for future expansion)
        for intent, handler in self.custom_handlers.items():
            if intent in cmd:
                return handler(cmd)

        return {
            "status": "unrecognized",
            "message": f"Command '{command_text}' not recognized. Try 'blue', 'red', 'clear', or 'save'."
        }

    def register_custom_action(self, keyword, handler_func):
        """Allow external Agentic AI modules to register custom actions."""
        self.custom_handlers[keyword.lower()] = handler_func
