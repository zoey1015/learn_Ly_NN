import torch
import math

class Attitude_mixer:
    """
    将位置控制器输出的三轴加速度命令转换为四旋翼的推力和期望姿态。
    
    物理模型 (ENU 坐标系，Z 轴向上):
        机体坐标系下，推力沿机体 Z 轴向上。
        惯性系下的加速度与姿态的关系：
            a_x = (U1/m) * (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))
            a_y = (U1/m) * (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))
            a_z = (U1/m) * cos(phi)*cos(theta) - g
        
        简化假设（小角度近似后的标准形式）：
            a_x ≈ (U1/m) * theta  (当 psi=0)
            a_y ≈ -(U1/m) * phi   (当 psi=0)
            a_z = (U1/m) - g
    """
    
    def __init__(self, mass: float = 0.486, gravity: float = 9.81):
        self.mass = mass
        self.gravity = gravity
        # 姿态角限幅（防止超过物理极限）
        self.max_tilt_angle = math.radians(45)  # 最大倾斜角 45°
    
    def compute(self, u_pos, psi_ref, gravity_compensated: bool = True):
        """
        将三轴加速度命令转换为推力和期望姿态。
        
        Args:
            u_pos: Tensor [Batch, 3] - 三轴控制输出 (u_x, u_y, u_z)
            psi_ref: Tensor [Batch, 1] - 偏航角参考值
            gravity_compensated: bool - u_z 是否已经包含重力补偿
                - True: u_z 来自 NN 控制器（如 Position_z_Dynamic 输出），已含重力
                - False: u_z 是纯运动学加速度 (ddot_z_desired)，需要加重力
        
        Returns:
            att_cmd: Tensor [Batch, 3] - 期望姿态 (phi_d, theta_d, psi_d)
            U1: Tensor [Batch, 1] - 总推力
        """
        device = u_pos.device
        dtype = u_pos.dtype
        
        ux = u_pos[:, 0:1]  # X 轴加速度命令
        uy = u_pos[:, 1:2]  # Y 轴加速度命令
        uz = u_pos[:, 2:3]  # Z 轴控制输出
        
        # 1. 确定 Z 轴总加速度（推力方向）
        if gravity_compensated:
            # u_z 已经是推力加速度 (u_z = U1/m)
            # 对于悬停：u_z ≈ g，对应 a_z = 0
            uz_total = uz
        else:
            # u_z 是期望的运动学加速度 (ddot_z)
            # 需要加上重力得到推力加速度
            uz_total = uz + self.gravity
        
        # 2. 计算总推力
        # U1/m = sqrt(ax^2 + ay^2 + (az+g)^2) = sqrt(ux^2 + uy^2 + uz_total^2)
        # 注意：ux, uy 是期望加速度，uz_total 是推力加速度
        acc_norm_sq = ux**2 + uy**2 + uz_total**2
        acc_norm = torch.sqrt(acc_norm_sq + 1e-8)  # 防止除零
        U1 = self.mass * acc_norm
        
        # 3. 姿态反解
        sin_psi = torch.sin(psi_ref)
        cos_psi = torch.cos(psi_ref)
        
        # Roll (phi) - 绕 X 轴旋转
        # 公式推导：sin(phi) = (ax*sin(psi) - ay*cos(psi)) / (U1/m)
        sin_phi = (ux * sin_psi - uy * cos_psi) / acc_norm
        sin_phi = torch.clamp(sin_phi, -0.999, 0.999)  # 数值稳定性
        phi_cmd = torch.asin(sin_phi)
        
        # Pitch (theta) - 绕 Y 轴旋转
        # 公式推导：tan(theta) = (ax*cos(psi) + ay*sin(psi)) / uz_total
        acc_planar_proj = ux * cos_psi + uy * sin_psi
        # 使用 atan2 处理符号，并确保 uz_total > 0 时 theta 的范围正确
        theta_cmd = torch.atan2(acc_planar_proj, torch.clamp(uz_total, min=0.1))
        
        # 4. 姿态角限幅（安全保护）
        phi_cmd = torch.clamp(phi_cmd, -self.max_tilt_angle, self.max_tilt_angle)
        theta_cmd = torch.clamp(theta_cmd, -self.max_tilt_angle, self.max_tilt_angle)
        
        # 5. 推力下限保护（防止负推力）
        U1 = torch.clamp(U1, min=0.1 * self.mass * self.gravity)
        
        # 组合输出：[Roll, Pitch, Yaw]
        att_cmd = torch.cat([phi_cmd, theta_cmd, psi_ref], dim=1)
        
        return att_cmd, U1
    
    def compute_inverse(self, att_cmd, U1):
        """
        逆映射：从推力和姿态反算期望加速度（用于验证）。
        
        Args:
            att_cmd: Tensor [Batch, 3] - 姿态 (phi, theta, psi)
            U1: Tensor [Batch, 1] - 总推力
        
        Returns:
            u_pos: Tensor [Batch, 3] - 对应的加速度命令 (ux, uy, uz_total)
        """
        phi = att_cmd[:, 0:1]
        theta = att_cmd[:, 1:2]
        psi = att_cmd[:, 2:3]
        
        # 推力加速度
        thrust_acc = U1 / self.mass
        
        # 反算加速度
        ux = thrust_acc * (torch.cos(phi) * torch.sin(theta) * torch.cos(psi) 
                          + torch.sin(phi) * torch.sin(psi))
        uy = thrust_acc * (torch.cos(phi) * torch.sin(theta) * torch.sin(psi) 
                          - torch.sin(phi) * torch.cos(psi))
        uz_total = thrust_acc * torch.cos(phi) * torch.cos(theta)
        
        return torch.cat([ux, uy, uz_total], dim=1)
    
    def get_hover_commands(self, batch_size: int = 1, device: str = 'cpu'):
        """
        获取悬停状态的命令（用于测试）。
        
        Returns:
            att_cmd: [0, 0, 0] - 水平姿态
            U1: m * g - 悬停推力
        """
        att_cmd = torch.zeros(batch_size, 3, device=device)
        U1 = torch.full((batch_size, 1), self.mass * self.gravity, device=device)
        return att_cmd, U1

