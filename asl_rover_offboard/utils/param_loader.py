import yaml
import os

from asl_rover_offboard.utils.param_types import (
    VehicleParams, ControlParams, Mission, LineTo, Straight, Arc, RoundedRectangle, RacetrackCapsule,
    Common, StartPose
)

class ParamLoader:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.params = self._load_yaml(yaml_path)

    def _load_yaml(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML file not found: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('/**', {}).get('ros__parameters', {})

    def _first(self, paths, default=None):
        """
        Return the first non-None from a list of nested key paths.
        Each path is a list of keys, e.g., ["mission","goal","x"].
        """
        for ks in paths:
            v = self.get_nested(ks, None)
            if v is not None:
                return v
        return default
    
    def get(self, key, default=None):
        """Generic access to a top-level parameter."""
        return self.params.get(key, default)

    def get_nested(self, keys, default=None):
        """Access nested parameter with list of keys."""
        ref = self.params
        try:
            for key in keys:
                ref = ref[key]
            return ref
        except (KeyError, TypeError):
            return default

    def get_topic(self, topic_key, default=None):
        """Access topic name from 'topics_names' section."""
        return self.get_nested(["topics_names", topic_key], default)

    def get_all_topics(self):
        """Return entire topics dictionary if available."""
        return self.get("topics_names", {})

    def get_control_gains(self):
        """Optional helper if using control_gains block."""
        return self.get("control_gains", {})

    def get_control_params(self) -> ControlParams:

        return ControlParams(
            frequency=self.get("control_parameters", {}).get("frequency", 50.0),
            N=self.get("control_parameters", {}).get("N", 25),
            v_max=self.get("control_parameters", {}).get("v_max", 1.0),
            w_max=self.get("control_parameters", {}).get("w_max", 1.5),
            Q=self.get("control_parameters", {}).get("Q", [5.0, 5.0, 1.0]),
            R=self.get("control_parameters", {}).get("R", [0.5, 0.3]),
            R_delta=self.get("control_parameters", {}).get("R_delta", [0.5, 0.3]),
            Qf_factor=self.get("control_parameters", {}).get("Qf_factor", 15.0)
        )
    
    def get_vehicle_params(self) -> VehicleParams:

        params = self.get("vehicle_parameters", {})

        required_fields = [
            'mass', 'base_width', 'input_scaling', 'wheel_radius', 'max_linear_speed',
            ('inertia', 'x'), ('inertia', 'y'), ('inertia', 'z'),
            'max_angular_speed', 'PWM_MIN', 'PWM_MAX', 'zero_position_armed', 
            ('omega_to_pwm_coefficient', 'x_2'), ('omega_to_pwm_coefficient', 'x_1'), ('omega_to_pwm_coefficient', 'x_0')
        ]

        for field in required_fields:
            if isinstance(field, tuple):
                group, subfield = field
                if group not in params or subfield not in params[group]:
                    raise ValueError(f"Missing required parameter: '{group}.{subfield}' in YAML file.")
            else:
                if field not in params:
                    raise ValueError(f"Missing required parameter: '{field}' in YAML file.")

        return VehicleParams(
            mass=params['mass'],
            base_width=params['base_width'],
            wheel_radius=params['wheel_radius'],
            inertia=[
                params['inertia']['x'],
                params['inertia']['y'],
                params['inertia']['z']
            ],
            max_linear_speed=params['max_linear_speed'],
            max_angular_speed=params['max_angular_speed'],
            PWM_MIN=params['PWM_MIN'],
            PWM_MAX=params['PWM_MAX'],
            input_scaling=params['input_scaling'],
            zero_position_armed=params['zero_position_armed'],
            omega_to_pwm_coefficient=[
                params['omega_to_pwm_coefficient']['x_2'],
                params['omega_to_pwm_coefficient']['x_1'],
                params['omega_to_pwm_coefficient']['x_0']
            ]
        )
    
    def get_mission(self) -> Mission:
        """Parse modern nested mission schema -> Mission variant."""
        mroot = self.get("mission", {}) or {}
        mtype = str(mroot.get("type", "line_to")).lower()
        common = mroot.get("common", {}) or {}
        start  = (common.get("start", {}) or {})
        params = mroot.get("params", {}) or {}

        c = Common(
            repeat=str(common.get("repeat", "none")).lower(),  # none|loop|pingpong
            start=StartPose(
                bool(start.get("use_current", True)),
                float(start.get("x", 0.0)),
                float(start.get("y", 0.0)),
                float(start.get("psi", 0.0)),
            ),
            speed=None if common.get("speed") is None else float(common["speed"]),
        )

        if mtype == "line_to":
            g = params.get("goal_xypsi", [0.0, 0.0, 0.0])
            return LineTo(common=c, goal_xypsi=(float(g[0]), float(g[1]), float(g[2])),
                          duration=float(params.get("duration", 0.0)))

        if mtype == "straight":
            return Straight(common=c, segment_distance=float(params.get("segment_distance", 0.0)))

        if mtype == "arc":
            ang = params.get("angle"); ang = None if ang is None else float(ang)
            yr  = params.get("yaw_rate"); yr = None if yr is None else float(yr)
            if ang is None and yr is None:
                raise ValueError("arc mission: provide one of {angle, yaw_rate}")
            return Arc(common=c, radius=float(params["radius"]), angle=ang, yaw_rate=yr,
                       cw=bool(params.get("cw", True)))

        if mtype == "rounded_rectangle":
            W = float(params["width"]); H = float(params["height"]); r = float(params["corner_radius"])
            if W <= 2*r or H <= 2*r:
                raise ValueError("rounded_rectangle: width/height must be > 2*corner_radius")
            return RoundedRectangle(common=c, width=W, height=H, corner_radius=r, cw=bool(params.get("cw", True)))

        if mtype == "racetrack_capsule":
            L = float(params["straight_length"]); r = float(params["radius"])
            if r <= 0.0:
                raise ValueError("racetrack_capsule: radius must be > 0")
            return RacetrackCapsule(common=c, straight_length=L, radius=r, cw=bool(params.get("cw", True)))

        raise ValueError(f"Unknown mission type: {mtype}")


    
    def as_dict(self):
        """Return full parameter dictionary."""
        return self.params
