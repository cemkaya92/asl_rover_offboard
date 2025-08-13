import yaml
import os

from asl_rover_offboard.utils.param_types import VehicleParams, ControlParams

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
    

    def as_dict(self):
        """Return full parameter dictionary."""
        return self.params
