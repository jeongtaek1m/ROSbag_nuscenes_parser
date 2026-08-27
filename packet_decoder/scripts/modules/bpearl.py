import sys
import os
import struct
import math
import argparse
import time as _time

from .utils import *

# ---------------------------------------------------------------------------
# RSBP Decoder
# ---------------------------------------------------------------------------
class RSBPDecoder:
    MSOP_LEN = 1248
    DIFOP_LEN = 1248
    MSOP_ID = b'\x55\xAA\x05\x0A\x5A\xA5\x50\xA0'
    DIFOP_ID = b'\xA5\xFF\x00\x5A\x11\x11\x55\x55'
    BLOCK_ID = b'\xFF\xEE'

    BLOCKS_PER_PKT = 12
    CHANNELS_PER_BLOCK = 32
    DISTANCE_MIN = 0.1
    DISTANCE_MAX = 150.0

    # Header V1 is 42 bytes
    HEADER_SIZE = 42
    # Block: id(2) + azimuth(2) + channels(32*3) = 100 bytes
    BLOCK_SIZE = 2 + 2 + 32 * 3

    # V3 defaults
    DISTANCE_RES = 0.005
    RX = 0.01473
    RY = 0.0085
    RZ = 0.09427
    BLOCK_DURATION = 55.52e-6

    FIRING_TSS_V3 = [
        0.00,  2.56,  5.12,  7.68, 10.24, 12.80, 15.36, 17.92,
        25.68, 28.24, 30.80, 33.36, 35.92, 38.48, 41.04, 43.60,
        1.28,  3.84,  6.40,  8.96, 11.52, 14.08, 16.64, 19.20,
        26.96, 29.52, 32.08, 34.64, 37.20, 39.76, 42.32, 44.88,
    ]

    FIRING_TSS_V4 = [
        0.00,  1.67,  3.34,  5.00,  6.67,  8.34, 10.01, 11.68,
        13.34, 15.01, 16.68, 18.35, 20.02, 21.68, 23.35, 25.02,
        26.69, 28.36, 30.02, 31.69, 33.36, 35.03, 36.70, 38.36,
        40.03, 41.70, 43.37, 45.04, 46.70, 48.37, 50.04, 51.71,
    ]

    def __init__(self):
        self.vert_angles = None
        self.horiz_angles = None
        self.user_chans = None
        self.calibration_ready = False
        self.reversal = False
        self.is_v4 = False
        self.first_pkt = True

        # Frame assembly state
        self.frame_points = []
        self.frame_ts = None

        self.block_az_diff = 20  # default

        # Set default firing times
        self._update_firing_params(self.FIRING_TSS_V3, 55.52)

    def _update_firing_params(self, firing_tss, blk_ts):
        self.chan_tss = [t * 1e-6 for t in firing_tss]
        self.chan_azis = [t / blk_ts for t in firing_tss]
        self.block_duration = blk_ts * 1e-6

    def decode_difop(self, data):
        """Parse RSBP DIFOP packet."""
        if len(data) < self.DIFOP_LEN:
            return
        if data[:8] != self.DIFOP_ID:
            return

        # RPM
        rpm = struct.unpack_from('>H', data, 8)[0]
        rps = rpm / 60.0 if rpm > 0 else 10.0
        if rps == 0:
            rps = 10.0
        self.block_az_diff = int(round(36000 * rps * self.block_duration))

        # Vertical angles: offset 468, 32 angles
        vert = parse_calibration_angles(data, 468, 32)
        # Horizontal angles: offset 564, 32 angles
        horiz = parse_calibration_angles(data, 564, 32)

        if vert is not None and horiz is not None:
            self.vert_angles = vert
            self.horiz_angles = horiz
            self.user_chans = gen_user_chan(vert)
            self.calibration_ready = True

        # Reversal flag: offset 337 (reserved_2[0])
        self.reversal = (data[337] != 0)

        # Return mode (offset 300)
        # 0x00 = dual, 0x01/0x02 = single

    def decode_msop(self, data, pkt_stamp, is_frame_begin=False):
        """Decode RSBP MSOP packet."""
        completed_frames = []

        if len(data) < self.MSOP_LEN:
            return completed_frames
        if data[:8] != self.MSOP_ID:
            return completed_frames
        if not self.calibration_ready:
            return completed_frames

        # Detect V4 on first packet (V1 header: lidar_type@31, lidar_model@32)
        if self.first_pkt:
            self.first_pkt = False
            lidar_type = data[31]
            lidar_model = data[32]
            if lidar_type == 0x03 and lidar_model == 0x04:
                self.is_v4 = True
                self.DISTANCE_RES = 0.0025
                self.RX = 0.01619
                self.RY = 0.0085
                self.RZ = 0.09571
                self._update_firing_params(self.FIRING_TSS_V4, 55.56)

        pkt_ts = pkt_stamp.to_sec()

        # Frame split: is_frame_begin set by C++ SDK (SplitStrategyByAngle, 0-deg crossing)
        if is_frame_begin and self.frame_points:
            completed_frames.append((self.frame_points, self.frame_ts))
            self.frame_points = []
            self.frame_ts = None

        for blk in range(self.BLOCKS_PER_PKT):
            blk_offset = self.HEADER_SIZE + blk * self.BLOCK_SIZE

            # Verify block ID
            if data[blk_offset:blk_offset + 2] != self.BLOCK_ID:
                break

            block_az = struct.unpack_from('>H', data, blk_offset + 2)[0]
            block_ts = pkt_ts + blk * self.block_duration

            for chan in range(self.CHANNELS_PER_BLOCK):
                chan_offset = blk_offset + 4 + chan * 3
                dist_raw = struct.unpack_from('>H', data, chan_offset)[0]
                intensity = data[chan_offset + 2]

                distance = dist_raw * self.DISTANCE_RES
                if distance < self.DISTANCE_MIN or distance > self.DISTANCE_MAX:
                    continue

                angle_vert = self.vert_angles[chan]
                angle_horiz = block_az + int(self.block_az_diff * self.chan_azis[chan])
                angle_horiz_final = angle_horiz + self.horiz_angles[chan]

                if self.reversal:
                    angle_horiz_final = 36000 - angle_horiz_final
                    angle_horiz = 36000 - angle_horiz

                cos_vert = math.cos(angle_vert * DEG_TO_RAD)
                sin_vert = math.sin(angle_vert * DEG_TO_RAD)
                cos_horiz_final = math.cos(angle_horiz_final * DEG_TO_RAD)
                sin_horiz_final = math.sin(angle_horiz_final * DEG_TO_RAD)
                cos_horiz = math.cos(angle_horiz * DEG_TO_RAD)
                sin_horiz = math.sin(angle_horiz * DEG_TO_RAD)

                x = distance * cos_vert * cos_horiz_final + self.RX * cos_horiz
                y = -distance * cos_vert * sin_horiz_final - self.RX * sin_horiz
                z = distance * sin_vert + self.RZ

                chan_ts = block_ts + self.chan_tss[chan]
                ring = self.user_chans[chan]

                self.frame_points.append((x, y, z, float(intensity), ring, chan_ts))
                if self.frame_ts is None:
                    self.frame_ts = chan_ts

        return completed_frames

    def flush(self):
        if self.frame_points:
            frame = (self.frame_points, self.frame_ts)
            self.frame_points = []
            self.frame_ts = None
            return [frame]
        return []
