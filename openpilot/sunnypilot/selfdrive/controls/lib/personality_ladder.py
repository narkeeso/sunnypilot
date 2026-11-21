"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.cereal import log

MPH_TO_MS = 0.44704
HYSTERESIS_MPH = 3.0
CONFIRM_FRAMES = 2

DEFAULT_STANDARD_MPH = 40
DEFAULT_RELAXED_MPH = 60


class PersonalityLadder:
  """Personality from speed: T_FOLLOW is a time headway, so the low tiers are cheap and the high tier is margin."""

  def __init__(self, standard_mph=DEFAULT_STANDARD_MPH, relaxed_mph=DEFAULT_RELAXED_MPH):
    self.tier = log.LongitudinalPersonality.aggressive
    self.candidate = None
    self.candidate_frames = 0
    self.set_thresholds(standard_mph, relaxed_mph)

  def set_thresholds(self, standard_mph, relaxed_mph):
    """Entry speeds in mph. Relaxed is kept above Standard, so a bad pair cannot invert the ladder."""
    self.standard_mph = max(1, int(standard_mph))
    self.relaxed_mph = max(self.standard_mph + 1, int(relaxed_mph))
    self.tiers = (
      (0.0, log.LongitudinalPersonality.aggressive),
      (self.standard_mph, log.LongitudinalPersonality.standard),
      (self.relaxed_mph, log.LongitudinalPersonality.relaxed),
    )

  def target(self, v_ego):
    """Highest tier entered at this speed, or the tier held while still inside its hysteresis band."""
    hysteresis = HYSTERESIS_MPH * MPH_TO_MS
    for threshold_mph, personality in reversed(self.tiers):
      threshold = threshold_mph * MPH_TO_MS
      if v_ego >= threshold or (personality == self.tier and v_ego >= threshold - hysteresis):
        return personality
    return self.tiers[0][1]

  def update(self, v_ego):
    """Commit a tier change after CONFIRM_FRAMES consecutive frames; returns the new tier or None."""
    target = self.target(v_ego)
    if target == self.tier:
      self.candidate = None
      self.candidate_frames = 0
      return None

    if target == self.candidate:
      self.candidate_frames += 1
    else:
      self.candidate = target
      self.candidate_frames = 1

    if self.candidate_frames < CONFIRM_FRAMES:
      return None

    self.tier = target
    self.candidate = None
    self.candidate_frames = 0
    return target

  def seed(self, v_ego):
    """Set the tier straight from speed, without walking the ladder (engage time)."""
    self.tier = max((p for threshold_mph, p in self.tiers if v_ego >= threshold_mph * MPH_TO_MS), default=self.tiers[0][1])
    self.candidate = None
    self.candidate_frames = 0
