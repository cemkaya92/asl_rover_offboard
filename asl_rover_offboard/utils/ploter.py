import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.t = []
        self.x = []; self.y = []; self.z = []
        self.vx = []; self.vy = []; self.vz = []
        self.roll = []; self.pitch = []; self.yaw = []
        self.x_cmd = []; self.y_cmd = []; self.z_cmd = []
        self.vx_cmd = []; self.vy_cmd = []; self.vz_cmd = []
        self.wl_cmd = []; self.wr_cmd = []; self.yaw_cmd = []


    def log(self, t_sim, pos, vel, rpy, pose_ref, u_mpc):
        self.t.append(t_sim)

        self.x.append(pos[0]); self.y.append(pos[1]); self.z.append(pos[2])
        self.vx.append(vel[0]); self.vy.append(vel[1]); self.vz.append(vel[2])
        self.roll.append(rpy[0]); self.pitch.append(rpy[1]); self.yaw.append(rpy[2])

        self.x_cmd.append(pose_ref[0]); self.y_cmd.append(pose_ref[1]); self.yaw_cmd.append(pose_ref[2])

        self.wl_cmd.append(u_mpc[0]); self.wr_cmd.append(u_mpc[1]); 

    def plot_logs(self):
        output_dir = os.path.expanduser("~/mpc_logs")
        os.makedirs(output_dir, exist_ok=True)
        t = self.t

        # Wheel Speed
        fig, axs = plt.subplots(1, 1, figsize=(10, 12), sharex=True)
        axs[0].plot(t, self.wl_cmd, label='Left Wheel Speed Cmd'); axs[0].plot(t, self.wr_cmd, label='Right Wheel Speed Cmd'); 
        axs[0].legend(); axs[0].grid(); axs[0].set_ylabel('Wheel Speed [rad/s]'); axs[0].set_xlabel('Time [s]')
        plt.suptitle("Wheel Speed"); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "wheel_speed_plot.png"))

        # Position
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axs[0].plot(t, self.x_cmd, label='X Cmd'); axs[0].plot(t, self.x, label='X'); axs[0].legend(); axs[0].grid(); axs[0].set_ylabel('X [m]')
        axs[1].plot(t, self.y_cmd, label='Y Cmd'); axs[1].plot(t, self.y, label='Y'); axs[1].legend(); axs[1].grid(); axs[1].set_ylabel('Y [m]')
        axs[2].plot(t, self.yaw_cmd, label='Yaw Cmd'); axs[2].plot(t, self.yaw, label='Yaw'); axs[2].legend(); axs[2].grid(); axs[2].set_ylabel('Yaw [rad]')
        plt.suptitle("Pose"); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "pose_plot.png"))

        # Velocity
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        axs[0].plot(t, self.vx, label='VX'); axs[0].legend(); axs[0].grid(); axs[0].set_ylabel('VX [m/s]')
        axs[1].plot(t, self.vy, label='VY'); axs[1].legend(); axs[1].grid(); axs[1].set_ylabel('VY [m/s]')
        plt.suptitle("Velocity"); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "velocity_plot.png"))

        print(f"[Logger] Logs saved to: {output_dir}")
