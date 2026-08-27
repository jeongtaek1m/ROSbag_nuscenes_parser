#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Writer: Chanwoo Kim
Date: 2026-03-30
Description: 
    Offline RoboSense packet-to-pointcloud decoder.

    Reads rslidar_msg/RslidarPacket messages from a ROS bag file,
    decodes them into sensor_msgs/PointCloud2 using the same logic
    as the C++ rs_driver SDK, and writes the results to a new bag.

    Supported lidar types: RSP128, RSM1, RSBP (V3/V4)
"""

import sys
import os
import struct
import math
import argparse
import json
import time as _time

import rosbag
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from modules import RSP128Decoder, RSBPDecoder, RSM1Decoder
from utils import ProgressBar

# ---------------------------------------------------------------------------
# PointCloud2 creation helper
# ---------------------------------------------------------------------------
POINT_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='ring', offset=16, datatype=PointField.UINT16, count=1),
    PointField(name='timestamp', offset=20, datatype=PointField.FLOAT64, count=1),
]
POINT_STEP = 28  # 4+4+4+4+2+2(pad)+8 -> actually let's compute: 4*4+2+2pad+8=28


def make_pointcloud2(points, stamp, frame_id):
    """Create a PointCloud2 message from a list of (x,y,z,intensity,ring,timestamp).

    Args:
        points: numpy array of shape (N, 6) with columns [x, y, z, intensity, ring, ts]
        stamp: rospy.Time
        frame_id: string
    """
    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    if len(points) == 0:
        msg.height = 1
        msg.width = 0
        msg.fields = POINT_FIELDS
        msg.is_bigendian = False
        msg.point_step = POINT_STEP
        msg.row_step = 0
        msg.data = b''
        msg.is_dense = True
        return msg

    n = len(points)
    buf = bytearray(n * POINT_STEP)

    for i, pt in enumerate(points):
        off = i * POINT_STEP
        struct.pack_into('<ffffH2xd', buf, off,
                         pt[0], pt[1], pt[2], pt[3],  # x,y,z,intensity
                         int(pt[4]),  # ring (uint16)
                         pt[5])  # timestamp (float64)

    msg.height = 1
    msg.width = n
    msg.fields = POINT_FIELDS
    msg.is_bigendian = False
    msg.point_step = POINT_STEP
    msg.row_step = POINT_STEP * n
    msg.data = bytes(buf)
    msg.is_dense = True
    return msg

# ---------------------------------------------------------------------------
# Decoder factory
# ---------------------------------------------------------------------------
def create_decoder(lidar_type):
    if lidar_type == 'RSP128':
        return RSP128Decoder()
    elif lidar_type == 'RSM1':
        return RSM1Decoder()
    elif lidar_type == 'RSBP':
        return RSBPDecoder()
    else:
        raise ValueError("Unsupported lidar type: {}".format(lidar_type))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_bag(input_bag_path, output_bag_path, config=None):
    """Main processing pipeline.

    Args:
        input_bag_path: path to input bag file
        output_bag_path: path to output bag file
        config: optional dict overriding LIDAR_CONFIG
    """
    if config is None:
        raise ValueError("Config is None")

    # Identify which packet topics exist in the bag
    bag_info = rosbag.Bag(input_bag_path, 'r')
    topic_info = bag_info.get_type_and_topic_info()
    topics_in_bag = topic_info.topics
    total_messages = sum(t.message_count for t in topics_in_bag.values())
    bag_info.close()

    active_topics = {}
    for cfg in config["lidars"]:
        pkt_topic = cfg["name"]
        if pkt_topic in topics_in_bag:
            active_topics[pkt_topic] = cfg
            print("[INFO] Found topic: {} ({})".format(pkt_topic, cfg['type']))
        else:
            print("[WARN] Topic not found in bag: {}".format(pkt_topic))

    if not active_topics:
        print("[ERROR] No matching packet topics found in the bag file.")
        return

    # Create decoders
    decoders = {}
    for pkt_topic, cfg in active_topics.items():
        decoders[pkt_topic] = create_decoder(cfg['type'])

    # -----------------------------------------------------------------------
    # Pass 1: Collect DIFOP packets for calibration
    # -----------------------------------------------------------------------
    print("\n[Pass 1] Collecting DIFOP packets for calibration...")
    pkt_topics = list(active_topics.keys())
    pkt_msg_count = sum(topics_in_bag[t].message_count for t in pkt_topics if t in topics_in_bag)
    pb1 = ProgressBar(pkt_msg_count, prefix='Pass 1')

    with rosbag.Bag(input_bag_path, 'r') as bag:
        all_ready = False
        for topic, msg, t in bag.read_messages(topics=pkt_topics):
            pb1.update()
            decoder = decoders[topic]
            if decoder.calibration_ready:
                # Check if all decoders are ready -> early exit
                if not all_ready:
                    all_ready = all(d.calibration_ready for d in decoders.values())
                    if all_ready:
                        break
                continue

            pkt_data = bytes(bytearray(msg.data))
            if msg.is_difop:
                decoder.decode_difop(pkt_data)
                if decoder.calibration_ready:
                    pb1.finish()
                    print("  [OK] Calibration loaded for {} ({})".format(
                        topic, active_topics[topic]['type']))
                    pb1 = ProgressBar(pkt_msg_count, prefix='Pass 1')
    pb1.finish()

    # Check calibration status
    for pkt_topic, decoder in decoders.items():
        if not decoder.calibration_ready:
            print("  [WARN] No DIFOP received for {}. "
                  "Points from this topic will be skipped.".format(pkt_topic))

    # -----------------------------------------------------------------------
    # Pass 2: Decode MSOP packets and write to output bag
    # -----------------------------------------------------------------------
    print("\n[Pass 2] Decoding packets and writing pointclouds...")

    frame_counts = {t: 0 for t in active_topics}
    point_counts = {t: 0 for t in active_topics}
    pb2 = ProgressBar(total_messages, prefix='Pass 2')

    with rosbag.Bag(input_bag_path, 'r') as in_bag, \
         rosbag.Bag(output_bag_path, 'w') as out_bag:

        for topic, msg, t in in_bag.read_messages():
            pb2.update()

            # Copy all original messages
            out_bag.write(topic, msg, t)

            # If it's a packet topic we care about, also decode it
            if topic in decoders:
                decoder = decoders[topic]
                pkt_data = bytes(bytearray(msg.data))

                if msg.is_difop:
                    decoder.decode_difop(pkt_data)
                else:
                    completed = decoder.decode_msop(pkt_data, t, bool(msg.is_frame_begin))
                    cfg = active_topics[topic]

                    for points, frame_ts in completed:
                        if not points:
                            continue

                        # Use frame timestamp for the message
                        if frame_ts is not None:
                            secs = int(frame_ts)
                            nsecs = int((frame_ts - secs) * 1e9)
                            stamp = rospy.Time(secs=secs, nsecs=nsecs)
                        else:
                            stamp = t

                        pc_msg = make_pointcloud2(
                            points, stamp, cfg['frame_id'])
                        out_bag.write(cfg['pc_topic'], pc_msg, stamp)

                        frame_counts[topic] += 1
                        point_counts[topic] += len(points)

        # Flush remaining frames
        for pkt_topic, decoder in decoders.items():
            cfg = active_topics[pkt_topic]
            for points, frame_ts in decoder.flush():
                if not points:
                    continue
                if frame_ts is not None:
                    secs = int(frame_ts)
                    nsecs = int((frame_ts - secs) * 1e9)
                    stamp = rospy.Time(secs=secs, nsecs=nsecs)
                else:
                    stamp = rospy.Time.now()

                pc_msg = make_pointcloud2(points, stamp, cfg['frame_id'])
                out_bag.write(cfg['pc_topic'], pc_msg, stamp)
                frame_counts[pkt_topic] += 1
                point_counts[pkt_topic] += len(points)

    pb2.finish()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print("Input:  {}".format(input_bag_path))
    print("Output: {}".format(output_bag_path))
    print("")
    for pkt_topic, cfg in active_topics.items():
        print("  {} ({})".format(pkt_topic, cfg['type']))
        print("    -> {} : {} frames, {} points".format(
            cfg['pc_topic'], frame_counts[pkt_topic], point_counts[pkt_topic]))
    print("")


def main():
    parser = argparse.ArgumentParser(
        description='Offline RoboSense packet-to-pointcloud decoder')
    parser.add_argument('input_bag', help='Path to input bag file with packet data')
    parser.add_argument('-o', '--output', default=None,
                        help='Path to output bag file (default: <input>_with_pc.bag)')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config', 'lidar_config.json')
    with open(config_path, 'r') as f:
        lidar_config = json.load(f)

    if not os.path.exists(args.input_bag):
        print("[ERROR] Input bag file not found: {}".format(args.input_bag))
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(args.input_bag)
        args.output = base + '_with_pc' + ext

    print("RoboSense Offline Packet Decoder")
    print("  Input:  {}".format(args.input_bag))
    print("  Output: {}".format(args.output))
    print("")

    process_bag(args.input_bag, args.output, config=lidar_config)


if __name__ == '__main__':
    main()
