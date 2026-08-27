import sys
import os
import struct
import math
import argparse
import time as _time

from .utils import *

# ---------------------------------------------------------------------------
# RSM1 Decoder
# ---------------------------------------------------------------------------
class RSM1Decoder:
    MSOP_LEN = 1210
    DIFOP_LEN = 256
    MSOP_ID = b'\x55\xAA\x5A\xA5'
    DIFOP_ID = b'\xA5\xFF\x00\x5A\x11\x11\x55\x55'

    BLOCKS_PER_PKT = 25
    CHANNELS_PER_BLOCK = 5
    DISTANCE_RES = 0.005
    DISTANCE_MIN = 0.2
    DISTANCE_MAX = 200.0

    HEADER_SIZE = 32
    # Block: time_offset(1) + return_seq(1) + 5 channels * 9 = 47 bytes
    BLOCK_SIZE = 1 + 1 + 5 * 9

    ANGLE_OFFSET = 32768
    SINGLE_PKT_NUM = 630
    FRAME_DURATION = 0.1

    def __init__(self):
        # RSM1 doesn't need DIFOP for calibration (absolute angles per point)
        self.calibration_ready = True

        # Frame assembly state
        self.frame_points = []
        self.frame_ts = None
        self.prev_pkt_seq = -1

    def decode_difop(self, _data):
        """RSM1 DIFOP: only used for echo mode. Calibration not needed."""
        pass

    def decode_msop(self, data, pkt_stamp, is_frame_begin=False):
        """Decode RSM1 MSOP packet."""
        completed_frames = []

        if len(data) < self.MSOP_LEN:
            return completed_frames
        if data[:4] != self.MSOP_ID:
            return completed_frames

        pkt_ts = pkt_stamp.to_sec()

        # Frame split: is_frame_begin set by C++ SDK (SplitStrategyBySeq, pkt_seq wraps to 0)
        if is_frame_begin and self.frame_points:
            completed_frames.append((self.frame_points, self.frame_ts))
            self.frame_points = []
            self.frame_ts = None

        for blk in range(self.BLOCKS_PER_PKT):
            blk_offset = self.HEADER_SIZE + blk * self.BLOCK_SIZE
            time_offset_us = data[blk_offset]  # microsecond offset
            point_time = pkt_ts + time_offset_us * 1e-6

            for chan in range(self.CHANNELS_PER_BLOCK):
                chan_offset = blk_offset + 2 + chan * 9
                dist_raw = struct.unpack_from('>H', data, chan_offset)[0]
                pitch_raw = struct.unpack_from('>H', data, chan_offset + 2)[0]
                yaw_raw = struct.unpack_from('>H', data, chan_offset + 4)[0]
                intensity = data[chan_offset + 6]

                distance = dist_raw * self.DISTANCE_RES
                if distance < self.DISTANCE_MIN or distance > self.DISTANCE_MAX:
                    continue

                pitch = pitch_raw - self.ANGLE_OFFSET  # signed 0.01 deg
                yaw = yaw_raw - self.ANGLE_OFFSET

                cos_pitch = math.cos(pitch * DEG_TO_RAD)
                x = distance * cos_pitch * math.cos(yaw * DEG_TO_RAD)
                y = distance * cos_pitch * math.sin(yaw * DEG_TO_RAD)
                z = distance * math.sin(pitch * DEG_TO_RAD)

                ring = chan

                self.frame_points.append((x, y, z, float(intensity), ring, point_time))
                if self.frame_ts is None:
                    self.frame_ts = point_time

        return completed_frames

    def flush(self):
        if self.frame_points:
            frame = (self.frame_points, self.frame_ts)
            self.frame_points = []
            self.frame_ts = None
            return [frame]
        return []