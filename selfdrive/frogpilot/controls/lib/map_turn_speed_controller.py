# PFEIFER - MTSC - Modified by FrogAi for FrogPilot
import json
import math

from openpilot.common.conversions import Conversions as CV
from openpilot.common.numpy_fast import clip

from openpilot.common.numpy_fast import interp

from openpilot.selfdrive.frogpilot.frogpilot_utilities import calculate_distance_to_point
from openpilot.selfdrive.frogpilot.frogpilot_variables import TO_RADIANS, params_memory

TARGET_JERK = -0.6   # m/s^3 should match up with the long planner
TARGET_ACCEL = -1.2  # m/s^2 should match up with the long planner
TARGET_OFFSET = 1.0  # seconds - This controls how soon before the curve you reach the target velocity. It also helps
                     # reach the target velocity when innacuracies in the distance modeling logic would cause overshoot.
                     # The value is multiplied against the target velocity to determine the additional distance. This is
                     # done to keep the distance calculations consistent but results in the offset actually being less
                     # time than specified depending on how much of a speed diffrential there is between v_ego and the
                     # target velocity.

def calculate_velocity(t, target_jerk, a_ego, v_ego):
  return v_ego + a_ego * t + target_jerk/2 * (t ** 2)

def calculate_distance(t, target_jerk, a_ego, v_ego):
  return t * v_ego + a_ego/2 * (t ** 2) + target_jerk/6 * (t ** 3)

class MapTurnSpeedController:
  def __init__(self):
    self.target_lat = 0.0
    self.target_lon = 0.0
    self.target_v = 0.0

  def target_speed(self, v_ego, a_ego, frogpilot_toggles) -> float:
    lat = 0.0
    lon = 0.0
    try:
      position = json.loads(params_memory.get("LastGPSPosition"))
      lat = position["latitude"]
      lon = position["longitude"]
    except:
      return 0.0

    try:
      target_velocities = json.loads(params_memory.get("MapTargetVelocities"))
    except:
      return 0.0

    min_dist = 1000
    min_idx = 0
    distances = []

    # find our location in the path
    for i in range(len(target_velocities)):
      target_velocity = target_velocities[i]
      tlat = target_velocity["latitude"]
      tlon = target_velocity["longitude"]
      d = calculate_distance_to_point(lat * TO_RADIANS, lon * TO_RADIANS, tlat * TO_RADIANS, tlon * TO_RADIANS)
      distances.append(d)
      if d < min_dist:
        min_dist = d
        min_idx = i

    # only look at values from our current position forward
    forward_points = target_velocities[min_idx:]
    forward_distances = distances[min_idx:]

    # find velocities that we are within the distance we need to adjust for
    valid_velocities = []
    for i in range(len(forward_points)):
      target_velocity = forward_points[i]
      tlat = target_velocity["latitude"]
      tlon = target_velocity["longitude"]
      tv = target_velocity["velocity"]
      if tv > v_ego:
        continue

      d = forward_distances[i]

      a_diff = (a_ego - TARGET_ACCEL)
      accel_t = abs(a_diff / TARGET_JERK)
      min_accel_v = calculate_velocity(accel_t, TARGET_JERK, a_ego, v_ego) / frogpilot_toggles.turn_aggressiveness

      max_d = 0
      if tv > min_accel_v:
        # calculate time needed based on target jerk
        a = 0.5 * TARGET_JERK
        b = a_ego
        c = v_ego - tv
        discriminant = (b**2 - 4 * a * c)
        if discriminant < 0:
          continue
        sqrt_disc = (discriminant ** 0.5)

        t_a = -1 * (sqrt_disc + b) / (2 * a)
        t_b = (sqrt_disc - b) / (2 * a)
        t = t_a if (t_a > 0 and not isinstance(t_a, complex)) else t_b
        if isinstance(t, complex) or t <= 0:
          continue

        max_d += calculate_distance(t, TARGET_JERK, a_ego, v_ego)
      else:
        t = accel_t
        max_d = calculate_distance(t, TARGET_JERK, a_ego, v_ego)

        # calculate additional time needed based on target accel
        t = abs((min_accel_v - tv) / TARGET_ACCEL)
        max_d += calculate_distance(t, 0, TARGET_ACCEL, min_accel_v)

      if d < (max_d + tv * TARGET_OFFSET) * frogpilot_toggles.curve_sensitivity:
        valid_velocities.append((float(tv), tlat, tlon))

    # Find the smallest velocity we need to adjust for
    min_v = 100.0
    target_lat = 0.0
    target_lon = 0.0
    for tv, lat, lon in valid_velocities:
      if tv < min_v:
        min_v = tv
        target_lat = lat
        target_lon = lon

    if self.target_v < min_v and not (self.target_lat == 0 and self.target_lon == 0):
      for i in range(len(forward_points)):
        target_velocity = forward_points[i]
        tlat = target_velocity["latitude"]
        tlon = target_velocity["longitude"]
        tv = target_velocity["velocity"]
        if tv > v_ego:
          continue

        if tlat == self.target_lat and tlon == self.target_lon and tv == self.target_v:
          return float(self.target_v)
      # not found so lets reset
      self.target_v = 0.0
      self.target_lat = 0.0
      self.target_lon = 0.0

    self.target_v = min_v
    self.target_lat = target_lat
    self.target_lon = target_lon

    # Scale the MTSC output from 2.0 m/s² lat accel to 3.2 m/s² lat accel
    if self.target_v > 0.1:  # avoid weird low-speed scaling
      self.target_v = self.target_v * math.sqrt(3.2 / 2.0)

    return self.target_v