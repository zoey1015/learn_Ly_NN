import torch
import numpy as np
import scipy

class Position_x_dynamic:
    """
    X position dynamics for a 3D quadrotor.
    state: x=[x,x_dot]
    control:u=Ux
    dynamic equation: x_ddot=Ux
    """
    def __init__(self, mass=0.486, gravity=9.81, *args, **kwargs):
        self.nx = 2  # state
        self.nu = 1  # control
        self.mass = mass      
        self.gravity = gravity 
    
    def forward(self, x, u):
        """
        计算连续时间动力学
        Args:
            x: shape (batch, 2) - [x, x_dot]
            u: shape (batch, 1) - Ux
            
        Returns:
            x_ddot: shape (batch, 1)
        """
        # 简化动力学: x_ddot = Ux
        # SecondOrderDiscreteTimeSystem expects only the acceleration term (qddot).
        acceleration = u
        return acceleration
    def linearized_dynamics(self, x, u):
        """
        返回 ∂ẋ/∂x 和 ∂ẋ/∂u
        
        对于这个系统（线性系统）:
        A = [[0, 1],
             [0, 0]]
        B = [[0],
             [1]]
        """
        if isinstance(x, np.ndarray):
            A = np.array([[0.0, 1.0],
                         [0.0, 0.0]])
            B = np.array([[0.0],
                         [1.0]])
            return A, B
        elif isinstance(x, torch.Tensor):
            dtype = x.dtype
            device = x.device
            A = torch.tensor([[0.0, 1.0],
                             [0.0, 0.0]], dtype=dtype, device=device)
            B = torch.tensor([[0.0],
                             [1.0]], dtype=dtype, device=device)
            return A, B
    
    def lqr_control(self, Q, R, x=None, u=None):
        """
        计算 LQR 增益
        控制律: u = K * (x - x*) + u*
        """
        x_np = self.x_equilibrium.numpy() if x is None else (
            x if isinstance(x, np.ndarray) else x.detach().numpy()
        )
        u_np = self.u_equilibrium.numpy() if u is None else (
            u if isinstance(u, np.ndarray) else u.detach().numpy()
        )
        
        A, B = self.linearized_dynamics(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K, S
    
    @property
    def x_equilibrium(self):
        """平衡点状态: [x=0, x_dot=0]"""
        return torch.zeros((2,))
    
    @property
    def u_equilibrium(self):
        """平衡点控制输入: Ux=0"""
        return torch.zeros((1,))


class PositionYDynamics:
    """
    Y position dynamics for a 3D quadrotor.
    state: x=[y,y_dot]
    control:u=Uy
    dynamic equation: y_ddot=Uy
    """

    def __init__(self, mass=0.486, gravity=9.81, *args, **kwargs):
        self.nx = 2  # 状态维度
        self.nu = 1  # 控制维度
        self.mass = mass  # 质量 (kg)
        self.gravity = gravity  # 重力加速度 (m/s^2)
        
    def forward(self, x, u):
        """
        计算连续时间动力学 (batched, pytorch)
        
        Args:
            x: shape (batch, 2) - [y, y_dot]
            u: shape (batch, 1) - Uy
            
        Returns:
            y_ddot: shape (batch, 1)
        """
        # 简化动力学: y_ddot = Uy
        # Return only acceleration for second-order integrator.
        acceleration = u
        return acceleration
    
    def linearized_dynamics(self, x, u):
        """
        返回 ∂ẋ/∂x 和 ∂ẋ/∂u
        
        对于这个系统（线性系统）:
        A = [[0, 1],
             [0, 0]]
        B = [[0],
             [1]]
        """
        if isinstance(x, np.ndarray):
            A = np.array([[0.0, 1.0],
                         [0.0, 0.0]])
            B = np.array([[0.0],
                         [1.0]])
            return A, B
        elif isinstance(x, torch.Tensor):
            dtype = x.dtype
            device = x.device
            A = torch.tensor([[0.0, 1.0],
                             [0.0, 0.0]], dtype=dtype, device=device)
            B = torch.tensor([[0.0],
                             [1.0]], dtype=dtype, device=device)
            return A, B
    
    def lqr_control(self, Q, R, x=None, u=None):
        """
        计算 LQR 增益
        控制律: u = K * (x - x*) + u*
        """
        x_np = self.x_equilibrium.numpy() if x is None else (
            x if isinstance(x, np.ndarray) else x.detach().numpy()
        )
        u_np = self.u_equilibrium.numpy() if u is None else (
            u if isinstance(u, np.ndarray) else u.detach().numpy()
        )
        
        A, B = self.linearized_dynamics(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K, S
    
    @property
    def x_equilibrium(self):
        """平衡点状态: [y=0, y_dot=0]"""
        return torch.zeros((2,))
    
    @property
    def u_equilibrium(self):
        """平衡点控制输入: Uy=0"""
        return torch.zeros((1,))


class PositionZDynamics:
    """
    Z position dynamics for a 3D quadrotor.
    state: x=[z,z_dot]
    control:u=Uz
    dynamic equation: z_ddot=Uz - g
    """

    def __init__(self, mass=0.486, gravity=9.81, *args, **kwargs):
        self.nx = 2  # 状态维度
        self.nu = 1  # 控制维度
        self.mass = mass  # 质量 (kg)
        self.gravity = gravity  # 重力加速度 (m/s^2)
        
    def forward(self, x, u):
        """
        计算连续时间动力学 (batched, pytorch)
        
        Args:
            x: shape (batch, 2) - [z, z_dot]
            u: shape (batch, 1) - Uz
            
        Returns:
            z_ddot: shape (batch, 1)
        """
        # 简化动力学: z_ddot = Uz - g
        # Return only acceleration for second-order integrator.
        acceleration = u - self.gravity
        return acceleration
    
    def linearized_dynamics(self, x, u):
        """
        返回 ∂ẋ/∂x 和 ∂ẋ/∂u
        
        对于这个系统（线性系统）:
        A = [[0, 1],
             [0, 0]]
        B = [[0],
             [1]]
        """
        if isinstance(x, np.ndarray):
            A = np.array([[0.0, 1.0],
                         [0.0, 0.0]])
            B = np.array([[0.0],
                         [1.0]])
            return A, B
        elif isinstance(x, torch.Tensor):
            dtype = x.dtype
            device = x.device
            A = torch.tensor([[0.0, 1.0],
                             [0.0, 0.0]], dtype=dtype, device=device)
            B = torch.tensor([[0.0],
                             [1.0]], dtype=dtype, device=device)
            return A, B
    
    def lqr_control(self, Q, R, x=None, u=None):
        """
        计算 LQR 增益
        控制律: u = K * (x - x*) + u*
        """
        x_np = self.x_equilibrium.numpy() if x is None else (
            x if isinstance(x, np.ndarray) else x.detach().numpy()
        )
        u_np = self.u_equilibrium.numpy() if u is None else (
            u if isinstance(u, np.ndarray) else u.detach().numpy()
        )
        
        A, B = self.linearized_dynamics(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K, S
    
    @property
    def x_equilibrium(self):
        """平衡点状态: [z=0, z_dot=0]"""
        return torch.zeros((2,))
    
    @property
    def u_equilibrium(self):
        """平衡点控制输入: Uz=g (悬停需要抵消重力)"""
        return torch.tensor([self.gravity])

if __name__ == "__main__":
    # 简单测试
    pos_x_dyn = Position_x_dynamic()
    x = torch.tensor([[0.0, 0.0]])
    u = torch.tensor([[1.0]])
    x_dot = pos_x_dyn.forward(x, u)
    print("Position X Dynamics:")
    print("x_dot:", x_dot)

    pos_y_dyn = PositionYDynamics()
    y_dot = pos_y_dyn.forward(x, u)
    print("Position Y Dynamics:")
    print("y_dot:", y_dot)

    pos_z_dyn = PositionZDynamics()
    z_dot = pos_z_dyn.forward(x, u)
    print("Position Z Dynamics:")
    print("z_dot:", z_dot)