"""Measure rosbridge sensor delivery without advertising or publishing commands."""

import argparse
import json
import math
import statistics
import threading
import time
from itertools import pairwise
from pathlib import Path

import roslibpy
import yaml

from .config import DEFAULT_HARDWARE_CONFIG
from ..hal.rosbridge import source_timestamp_ns


def summarise(records):
    """Compare arrival intervals and source stamps without assuming synced clocks."""
    encoders = [row for row in records if row['stream'] == 'encoders']
    if not encoders:
        return []
    origin = encoders[0]
    result = []
    for phase in ('encoders_only_before', 'all_streams', 'encoders_only_after'):
        for stream in ('encoders', 'camera', 'lidar'):
            rows = [row for row in records if row['phase'] == phase and row['stream'] == stream]
            if len(rows) < 2:
                continue
            arrivals = [b['received_s']-a['received_s'] for a, b in pairwise(rows)]
            stamps = [(b['stamp_ns']-a['stamp_ns'])/1e9 for a, b in pairwise(rows)]
            ages = [(row['received_s']-origin['received_s'])
                    -(row['stamp_ns']-origin['stamp_ns'])/1e9 for row in rows]
            elapsed = rows[-1]['received_s']-rows[0]['received_s']
            result.append({
                'phase': phase, 'stream': stream, 'messages': len(rows),
                'arrival_hz': (len(rows)-1)/elapsed if elapsed > 0 else None,
                'arrival_gap_median_s': statistics.median(arrivals),
                'arrival_gap_max_s': max(arrivals),
                'source_gap_median_s': statistics.median(stamps),
                'source_gap_max_s': max(stamps),
                'repeated_stamps': sum(gap == 0 for gap in stamps),
                'backward_stamps': sum(gap < 0 for gap in stamps),
                'relative_age_median_s': statistics.median(ages),
                'relative_age_max_s': max(ages),
                'relative_age_change_s': ages[-1]-ages[0],
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', help='Robot IP address on the local robot network.')
    parser.add_argument('--config', type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument('--seconds', type=float, default=10., help='Duration of EACH of three phases.')
    parser.add_argument('--output', type=Path, default=Path('outputs/hardware_timing.json'))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive and finite')
    connection = yaml.safe_load(args.config.read_text())['connection']
    client = roslibpy.Ros(host=args.host, port=connection['port'])
    topics, records, errors = {}, [], []
    lock = threading.Lock()
    phase = 'startup'
    started = time.monotonic()

    def receive(stream):
        def callback(message):
            received = time.monotonic()-started
            try:
                stamp = source_timestamp_ns(message)
            except ValueError as exc:
                with lock:
                    if len(errors) < 20:
                        errors.append(f'{stream}: {exc}')
                return
            with lock:
                records.append({'stream': stream, 'phase': phase,
                                'received_s': received, 'stamp_ns': stamp})
        return callback

    def subscribe(stream, key, message_type):
        topic = roslibpy.Topic(client, connection['topics'][key], message_type)
        topics[stream] = topic
        topic.subscribe(receive(stream))

    try:
        client.run(timeout=5)
        print('Connected. Read-only sensor subscriptions; no robot commands.', flush=True)
        phase = 'encoders_only_before'
        subscribe('encoders', 'joint_states', 'sensor_msgs/msg/JointState')
        print(f'Joint states alone: {args.seconds:g}s', flush=True)
        time.sleep(args.seconds)
        phase = 'all_streams'
        subscribe('camera', 'image', 'sensor_msgs/msg/CompressedImage')
        subscribe('lidar', 'laser_scan', 'sensor_msgs/msg/LaserScan')
        print(f'Joint states, camera and LiDAR: {args.seconds:g}s', flush=True)
        time.sleep(args.seconds)
        for name in ('camera', 'lidar'):
            topics.pop(name).unsubscribe()
        phase = 'encoders_only_after'
        print(f'Joint states alone again: {args.seconds:g}s', flush=True)
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        errors.append('Interrupted; report contains partial measurements.')
    except Exception as exc:  # noqa: BLE001 - save diagnostic evidence even on connection failure
        errors.append(f'{type(exc).__name__}: {exc}')
    finally:
        try:
            if client.is_connected:
                for topic in topics.values():
                    topic.unsubscribe()
        finally:
            client.terminate()
            with lock:
                snapshot, problems = records.copy(), errors.copy()
            if not any(row['stream'] == 'encoders' for row in snapshot):
                problems.append('No encoder messages received; check connection, server and topic.')
            report = {
                'host': args.host, 'port': connection['port'], 'topics': connection['topics'],
                'seconds_per_phase': args.seconds,
                'age_reference': 'First encoder arrival; relative ages are NOT absolute network latency.',
                'summary': summarise(snapshot), 'errors': problems, 'samples': snapshot,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'summary': report['summary'], 'errors': problems}, indent=2))
            print(f'Saved report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
