import sys
import os
import struct
import math
import argparse
import time as _time

from .utils import *

# ---------------------------------------------------------------------------
# RSP128 Decoder
# ---------------------------------------------------------------------------
class RSP128Decoder:
    MSOP_LEN = 1248
    DIFOP_LEN = 1248
    MSOP_ID = b'\x55\xAA\x05\x5A'
    DIFOP_ID = b'\xA5\xFF\x00\x5A\x11\x11\x55\x55'
    BLOCK_ID = 0xFE

    BLOCKS_PER_PKT = 3
    CHANNELS_PER_BLOCK = 128
    DISTANCE_RES = 0.005
    DISTANCE_MIN = 0.4
    DISTANCE_MAX = 250.0

    # Header V2 is 80 bytes
    HEADER_SIZE = 80
    # Block: id(1) + ret_id(1) + azimuth(2) + channels(128*3) = 388 bytes
    BLOCK_SIZE = 1 + 1 + 2 + 128 * 3

    # Lens center offsets
    RX = 0.02892
    RZ = 0.0

    # Block duration in seconds
    BLOCK_DURATION = 55.56e-6

    # Firing time offsets (microseconds) - from decoder_RSP128.hpp
    FIRING_TSS = [
        0.0,    0.0,    0.0,    0.0,    1.13,   1.13,   1.13,   1.13,
        2.13,   2.13,   2.13,   2.13,   3.26,   3.26,   3.26,   3.26,
        4.26,   4.26,   4.26,   4.26,   5.38,   5.38,   5.38,   5.38,
        6.38,   6.38,   6.38,   6.38,   7.51,   7.51,   7.51,   7.51,
        8.51,   8.51,   8.51,   8.51,   10.01,  10.01,  10.01,  10.01,
        11.38,  11.38,  11.38,  11.38,  13.31,  13.31,  13.31,  13.31,
        15.11,  15.11,  15.11,  15.11,  17.04,  17.04,  17.04,  17.04,
        18.85,  18.85,  18.85,  18.85,  21.14,  21.14,  21.14,  21.14,
        21.14,  23.31,  23.31,  23.31,  23.31,  25.61,  25.61,  25.61,
        25.61,  27.78,  27.78,  27.78,  27.78,  30.07,  30.07,  30.07,
        30.07,  32.24,  32.24,  32.24,  32.24,  34.54,  34.54,  34.54,
        34.54,  36.7,   36.7,   36.7,   36.7,   38.64,  38.64,  38.64,
        38.64,  40.44,  40.44,  40.44,  40.44,  41.94,  41.94,  41.94,
        41.94,  43.3,   43.3,   43.3,   43.3,   44.8,   44.8,   44.8,
        44.8,   46.17,  46.17,  46.17,  46.17,  47.66,  47.66,  47.66,
        47.66,  49.03,  49.03,  49.03,  49.03,  50.53,  50.53,  53.771,
    ]

    CHAN_TSS = [t * 1e-6 for t in FIRING_TSS]
    CHAN_AZIS = [t / 55.56 for t in FIRING_TSS]

    def __init__(self):
        self.vert_angles = None   # list of int32 (0.01 deg units)
        self.horiz_angles = None  # list of int32 (0.01 deg units)
        self.user_chans = None
        self.calibration_ready = False

        # Frame assembly state
        self.frame_points = []
        self.frame_ts = None  # timestamp of first point in frame

        # Block azimuth diff (updated from DIFOP rpm)
        self.block_az_diff = 20  # default

    def decode_difop(self, data):
        """Parse DIFOP packet for calibration angles."""
        if len(data) < self.DIFOP_LEN:
            return
        if data[:8] != self.DIFOP_ID:
            return

        # RPM -> rps -> block_az_diff 
        # Read big-endian 2 bytes from offset 8
        rpm = struct.unpack_from('>H', data, 8)[0]
        rps = rpm / 60.0 if rpm > 0 else 10.0 # default: 10.0 rps
        if rps == 0:
            rps = 10.0
        self.block_az_diff = int(round(36000 * rps * self.BLOCK_DURATION))

        # Vertical angles: offset 468, 128 angles
        vert = parse_calibration_angles(data, 468, 128)
        # Horizontal angles: offset 852, 128 angles
        horiz = parse_calibration_angles(data, 852, 128)

        if vert is not None and horiz is not None:
            self.vert_angles = vert
            self.horiz_angles = horiz
            self.user_chans = gen_user_chan(vert)
            self.calibration_ready = True

    def decode_msop(self, data, pkt_stamp, is_frame_begin=False):
        """Decode MSOP packet. Returns list of completed frames as
        (points_list, rospy.Time) tuples.
        """
        completed_frames = []

        if len(data) < self.MSOP_LEN:
            return completed_frames
        if data[:4] != self.MSOP_ID:
            return completed_frames
        if not self.calibration_ready:
            return completed_frames

        pkt_ts = pkt_stamp.to_sec()

        # Frame split: is_frame_begin set by C++ SDK (SplitStrategyByAngle, 0-deg crossing)
        if is_frame_begin and self.frame_points:
            # Use pkt_ts (0-deg crossing time) as the frame stamp,
            # which is the conventional reference used by rs_driver.
            completed_frames.append((self.frame_points, pkt_ts))
            self.frame_points = []
            self.frame_ts = None

        for blk in range(self.BLOCKS_PER_PKT):
            blk_offset = self.HEADER_SIZE + blk * self.BLOCK_SIZE

            # Verify block ID
            if data[blk_offset] != self.BLOCK_ID:
                break

            block_az = struct.unpack_from('>H', data, blk_offset + 2)[0]
            block_ts = pkt_ts + blk * self.BLOCK_DURATION

            # Parse channels
            for chan in range(self.CHANNELS_PER_BLOCK):
                chan_offset = blk_offset + 4 + chan * 3
                dist_raw = struct.unpack_from('>H', data, chan_offset)[0]
                intensity = data[chan_offset + 2]

                distance = dist_raw * self.DISTANCE_RES

                if distance < self.DISTANCE_MIN or distance > self.DISTANCE_MAX:
                    continue

                angle_vert = self.vert_angles[chan]
                angle_horiz = block_az + int(self.block_az_diff * self.CHAN_AZIS[chan])
                angle_horiz_final = angle_horiz + self.horiz_angles[chan]

                cos_vert = math.cos(angle_vert * DEG_TO_RAD)
                sin_vert = math.sin(angle_vert * DEG_TO_RAD)
                cos_horiz_final = math.cos(angle_horiz_final * DEG_TO_RAD)
                sin_horiz_final = math.sin(angle_horiz_final * DEG_TO_RAD)
                cos_horiz = math.cos(angle_horiz * DEG_TO_RAD)
                sin_horiz = math.sin(angle_horiz * DEG_TO_RAD)

                x = distance * cos_vert * cos_horiz_final + self.RX * cos_horiz
                y = -distance * cos_vert * sin_horiz_final - self.RX * sin_horiz
                z = distance * sin_vert + self.RZ

                chan_ts = block_ts + self.CHAN_TSS[chan]
                ring = self.user_chans[chan]

                self.frame_points.append((x, y, z, float(intensity), ring, chan_ts))
                if self.frame_ts is None:
                    self.frame_ts = chan_ts

        return completed_frames

    def flush(self):
        """Flush remaining points as a final frame.

        Stamped like every other frame (see decode_msop): the reference is the
        end of the sweep, so use the last point's time. self.frame_ts holds the
        *first* point's time and would put this final frame on a different
        convention from all the others.
        """
        if self.frame_points:
            last_ts = self.frame_points[-1][5]
            frame = (self.frame_points, last_ts)
            self.frame_points = []
            self.frame_ts = None
            return [frame]
        return []
