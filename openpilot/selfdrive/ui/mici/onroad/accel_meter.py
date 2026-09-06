import numpy as np
import pyray as rl
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.widgets import Widget


class AccelMeter(Widget):
  ACCEL_THRESHOLD = 0.05  # m/s^2, throttle above this
  BAR_WIDTH = 24
  BAR_HEIGHT = 320
  TOP_OFFSET = 64  # below the lead glyph

  def __init__(self):
    super().__init__()

  def _render(self, _):
    center_x = self._rect.x + self._rect.width - SIDE_PANEL_WIDTH / 2
    top_y = self._rect.y + self.TOP_OFFSET

    rl.draw_rectangle_rec(rl.Rectangle(center_x - self.BAR_WIDTH / 2, top_y,
                                       self.BAR_WIDTH, self.BAR_HEIGHT),
                          rl.Color(255, 255, 255, 38))

    if not (ui_state.status == UIStatus.ENGAGED and ui_state.started):
      return

    if ui_state.sm['carState'].brakePressed:
      color = rl.Color(255, 0, 21, 220)
    elif ui_state.sm['longitudinalPlan'].aTarget > self.ACCEL_THRESHOLD:
      color = rl.Color(0, 255, 64, 220)
    else:
      color = rl.Color(255, 200, 0, 220)

    rl.draw_rectangle_rec(rl.Rectangle(center_x - self.BAR_WIDTH / 2, top_y,
                                       self.BAR_WIDTH, self.BAR_HEIGHT),
                          color)