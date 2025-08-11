# laser_scan_sector_filter.py
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

def wrap_2pi(x):
    return np.mod(x, 2.0*np.pi)

class LaserScanSectorFilter(Node):
    def __init__(self):
        super().__init__('laser_scan_sector_filter')

        # Params
        self.declare_parameter('input_topic', '/merged_scan')
        self.declare_parameter('output_topic', '/scan_filtered')
        self.declare_parameter('min_angle_deg', -65.0)     # degrees
        self.declare_parameter('max_angle_deg',  65.0)     # degrees
        self.declare_parameter('mode', 'mask')             # 'mask' or 'crop'

        self.input_topic  = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.mode         = self.get_parameter('mode').get_parameter_value().string_value

        self.min_angle = math.radians(
            self.get_parameter('min_angle_deg').get_parameter_value().double_value)
        self.max_angle = math.radians(
            self.get_parameter('max_angle_deg').get_parameter_value().double_value)

        # Allow updating angles at runtime
        self.add_on_set_parameters_callback(self._on_param_update)

        self.pub = self.create_publisher(LaserScan, self.output_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.input_topic, self._cb, 10)

        self.get_logger().info(
            f"Filtering {self.input_topic} -> {self.output_topic} | "
            f"sector [{math.degrees(self.min_angle):.1f}°, {math.degrees(self.max_angle):.1f}°] | mode={self.mode}"
        )

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'min_angle_deg':
                self.min_angle = math.radians(p.value)
            elif p.name == 'max_angle_deg':
                self.max_angle = math.radians(p.value)
            elif p.name == 'mode':
                self.mode = p.value
        return rclpy.parameter.SetParametersResult(successful=True)

    def _cb(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0 or msg.angle_increment == 0.0:
            return

        # Build angle array for all beams
        angles = msg.angle_min + msg.angle_increment * np.arange(n, dtype=float)

        # Normalize to [0, 2π) so we can do circular comparisons robustly
        a_all = wrap_2pi(angles)
        a_min = wrap_2pi(self.min_angle)
        a_max = wrap_2pi(self.max_angle)

        if a_min <= a_max:
            mask = (a_all >= a_min) & (a_all <= a_max)
        else:
            # wrap-around sector (e.g., 300°..30°)
            mask = (a_all >= a_min) | (a_all <= a_max)

        if not np.any(mask):
            # Nothing in sector — drop silently or warn once
            self.get_logger().warn("No scan samples fall inside the requested sector.")
            return

        if self.mode == 'mask':
            out = LaserScan()
            out.header = msg.header
            out.angle_min = msg.angle_min
            out.angle_max = msg.angle_max
            out.angle_increment = msg.angle_increment
            out.time_increment = msg.time_increment
            out.scan_time = msg.scan_time
            out.range_min = msg.range_min
            out.range_max = msg.range_max

            ranges = np.asarray(msg.ranges, dtype=float)
            intens = np.asarray(msg.intensities, dtype=float) if msg.intensities else None

            # set outside-sector to +inf
            ranges[~mask] = float('inf')
            out.ranges = ranges.tolist()
            out.intensities = intens.tolist() if intens is not None else []

            self.pub.publish(out)
            return

        # mode == 'crop'
        idx = np.where(mask)[0]
        # Ensure increasing physical angle in the output
        # If the sector wraps, reorder so it starts from the requested min angle
        # Find nearest index to min_angle on the circle:
        # project desired min onto the input grid
        desired_start = self.min_angle
        # compute circular distance; pick the masked index with smallest angular distance to desired start
        circ_dist = np.minimum(
            np.mod(angles[idx] - desired_start, 2*np.pi),
            np.mod(desired_start - angles[idx], 2*np.pi)
        )
        start_pos = int(idx[np.argmin(circ_dist)])
        # Build contiguous order starting at start_pos, respecting scan order and wrapping
        order = list(range(start_pos, n)) + list(range(0, start_pos))
        order = np.array(order, dtype=int)
        order = order[np.isin(order, idx)]  # keep only indices in sector, in order

        out = LaserScan()
        out.header = msg.header
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max

        out.ranges = np.asarray(msg.ranges, dtype=float)[order].tolist()
        if msg.intensities:
            out.intensities = np.asarray(msg.intensities, dtype=float)[order].tolist()
        else:
            out.intensities = []

        out.angle_min = float(angles[order[0]])
        out.angle_max = float(angles[order[-1]])

        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(LaserScanSectorFilter())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
