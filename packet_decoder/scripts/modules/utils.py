import sys
import os
import struct
import math
import argparse
import json
import calendar
import time as _time

# ---------------------------------------------------------------------------
# Trigonometric helpers (angles in 0.01-degree units, i.e. 36000 = 360 deg)
# ---------------------------------------------------------------------------
DEG_TO_RAD = math.pi / 18000.0  # 0.01 degree -> radian

# ---------------------------------------------------------------------------
# Timestamp parsing (matches basic_attr.hpp)
# ---------------------------------------------------------------------------
def parse_timestamp_utc_us(data, offset):
    """Parse RSTimestampUTC (10 bytes) -> microseconds since epoch."""
    sec = 0
    for i in range(6):
        sec = (sec << 8) | data[offset + i]
    us = 0
    for i in range(4):
        us = (us << 8) | data[offset + 6 + i]
    return sec * 1000000 + us


def parse_timestamp_ymd(data, offset):
    """Parse RSTimestampYMD (10 bytes) -> microseconds since epoch."""

    year = data[offset] + 2000
    month = data[offset + 1]
    day = data[offset + 2]
    hour = data[offset + 3]
    minute = data[offset + 4]
    second = data[offset + 5]
    ms = struct.unpack_from('>H', data, offset + 6)[0]
    us = struct.unpack_from('>H', data, offset + 8)[0]

    # mktime expects local time; use calendar.timegm for UTC
    t = (year, month, day, hour, minute, second, 0, 0, -1)
    sec = int(calendar.timegm(t))
    return sec * 1000000 + ms * 1000 + us


# ---------------------------------------------------------------------------
# Calibration angle parsing (matches chan_angles.hpp)
# ---------------------------------------------------------------------------
def parse_calibration_angles(data, offset, count):
    """Parse RSCalibrationAngle[count] from DIFOP packet.

    Returns list of int32 angles in 0.01-degree units.
    Returns None if calibration is invalid.
    """
    angles = []
    for i in range(count):
        # angle information is 3 bytes (sign(1) + value(2))
        base = offset + i * 3
        sign = data[base] # 0: positive, 1: negative
        value = struct.unpack_from('>H', data, base + 1)[0] # value is big-endian 2 bytes
        if sign == 0xFF:
            return None  # invalid calibration
        if sign != 0:
            value = -value
        angles.append(value)
    return angles


def gen_user_chan(vert_angles):
    """Generate user channel mapping sorted by vertical angle (same as C++)."""
    user_chans = []
    for angle in vert_angles:
        chan = sum(1 for a in vert_angles if a < angle)
        user_chans.append(chan)
    return user_chans