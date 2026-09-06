import numpy as np
import pyray as rl
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import Widget


class LeadIndicator(Widget):
  CLOSE_DIST = 30.0  # m, red when closer

  def __init__(self):
    super().__init__()

  def _render(self, _):
    if not ui_state.started:
      return

    radar_state = ui_state.sm['radarState'] if ui_state.sm.valid['radarState'] else None
    lead = radar_state.leadOne if radar_state is not None else None
    if lead is None or not lead.present:
      return

    center_x = self._rect.x + self._rect.width - SIDE_PANEL_WIDTH / 2
    y = self._rect.y + 24

    close = lead.dRel < self.CLOSE_DIST
    status_color = rl.Color(255, 60, 60, 255) if close else rl.Color(218, 202, 37, 255)

    bw, bh = 56, 20
    bx = center_x - bw / 2
    by = y - bh / 2

    rl.draw_rectangle_rec(rl.Rectangle(bx - 2, by - 2, bw + 4, bh + 4), rl.BLACK)
    rl.draw_rectangle_rec(rl.Rectangle(bx, by, bw, bh), rl.WHITE)
    rl.draw_rectangle_rec(rl.Rectangle(bx + 10, by - 6, bw - 20, 14), rl.WHITE)
    rl.draw_rectangle_rec(rl.Rectangle(bx + 3, by + bh - 3, 9, 3), status_color)
    rl.draw_rectangle_rec(rl.Rectangle(bx + bw - 12, by + bh - 3, 9, 3), status_color)