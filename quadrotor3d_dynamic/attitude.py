import torch
import numpy as np
import scipy

class AttitudeDynamics:
    """
    attiyude dynamic
    """
        def __init__(
        self, 
        length=0.25, 
        mass=0.486, 
        Ixx=0.00383, 
        Iyy=0.00383, 
        Izz=0.00765,
        *args, 
        **kwargs
    ):
        self.nx = 6  # 状态维度
        self.nu = 3  # 控制维度
        
        # 物理参数
        self.length = length      # 机臂长度 (m)
        self.mass = mass          # 质量 (kg)
        self.Ixx = Ixx            # x轴转动惯量 (kg·m²)
        self.Iyy = Iyy            # y轴转动惯量 (kg·m²)
        self.Izz = Izz            # z轴转动惯量 (kg·m²)
        
    def forward(self, x, u):
        """
        计算连续时间动力学 (batched, pytorch)
        这是将被 auto_LiRPA 用于边界计算的实际函数
        
        Args:
            x: shape (batch, 6) - [φ, θ, ψ, φ̇, θ̇, ψ̇]
            u: shape (batch, 3) - [τx, τy, τz]
            
        Returns:
            x_dot: shape (batch, 6) - [φ̇, θ̇, ψ̇, φ̈, θ̈, ψ̈]
        """
        # 提取状态
        phi = x[:, 0:1]      # 滚转角
        theta = x[:, 1:2]    # 俯仰角
        psi = x[:, 2:3]      # 偏航角
        p = x[:, 3:4]        # 滚转角速度
        q = x[:, 4:5]        # 俯仰角速度
        r = x[:, 5:6]        # 偏航角速度
        
        # 提取控制输入
        tau_x = u[:, 0:1]
        tau_y = u[:, 1:2]
        tau_z = u[:, 2:3]
        
        # ========== 1. 欧拉角速率方程 ==========
        # [φ̇]   [1  sin(φ)tan(θ)  cos(φ)tan(θ)] [p]
        # [θ̇] = [0     cos(φ)       -sin(φ)   ] [q]
        # [ψ̇]   [0  sin(φ)/cos(θ)  cos(φ)/cos(θ)] [r]
        
        phi_dot = p + torch.sin(phi) * torch.tan(theta) * q + torch.cos(phi) * torch.tan(theta) * r
        theta_dot = torch.cos(phi) * q - torch.sin(phi) * r
        psi_dot = (torch.sin(phi) / torch.cos(theta)) * q + (torch.cos(phi) / torch.cos(theta)) * r
        
        # ========== 2. 角加速度方程（欧拉动力学方程）==========
        # I * ω̇ = τ - ω × (I*ω)
        # 展开后:
        # ṗ = τx/Ixx + (Iyy - Izz)/Ixx * q*r
        # q̇ = τy/Iyy + (Izz - Ixx)/Iyy * p*r
        # ṙ = τz/Izz + (Ixx - Iyy)/Izz * p*q
        
        p_dot = tau_x / self.Ixx + ((self.Iyy - self.Izz) / self.Ixx) * q * r
        q_dot = tau_y / self.Iyy + ((self.Izz - self.Ixx) / self.Iyy) * p * r
        r_dot = tau_z / self.Izz + ((self.Ixx - self.Iyy) / self.Izz) * p * q
        
        # 返回状态导数
        return torch.cat((phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot), dim=1)
    
    def linearized_dynamics(self, x, u):
        """
        返回 ∂ẋ/∂x 和 ∂ẋ/∂u
        
        在平衡点 (φ=0, θ=0, ψ=0, p=0, q=0, r=0) 线性化
        """
        if isinstance(x, np.ndarray):
            # A 矩阵 (6x6)
            A = np.zeros((6, 6))
            
            # 上半部分：角速度到角度的映射
            # φ̇ = p (线性化后)
            # θ̇ = q
            # ψ̇ = r
            A[0, 3] = 1.0  # φ̇ = p
            A[1, 4] = 1.0  # θ̇ = q
            A[2, 5] = 1.0  # ψ̇ = r
            
            # 下半部分：角加速度（在平衡点附近，交叉项为0）
            # ṗ = τx/Ixx
            # q̇ = τy/Iyy
            # ṙ = τz/Izz
            # (线性化后，p*q, q*r, p*r 项都为0)
            
            # B 矩阵 (6x3)
            B = np.zeros((6, 3))
            B[3, 0] = 1.0 / self.Ixx  # ṗ = τx/Ixx
            B[4, 1] = 1.0 / self.Iyy  # q̇ = τy/Iyy
            B[5, 2] = 1.0 / self.Izz  # ṙ = τz/Izz
            
            return A, B
            
        elif isinstance(x, torch.Tensor):
            dtype = x.dtype
            device = x.device
            
            # A 矩阵 (6x6)
            A = torch.zeros((6, 6), dtype=dtype, device=device)
            A[0, 3] = 1.0
            A[1, 4] = 1.0
            A[2, 5] = 1.0
            
            # B 矩阵 (6x3)
            B = torch.zeros((6, 3), dtype=dtype, device=device)
            B[3, 0] = 1.0 / self.Ixx
            B[4, 1] = 1.0 / self.Iyy
            B[5, 2] = 1.0 / self.Izz
            
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
        """平衡点状态: [φ=0, θ=0, ψ=0, φ̇=0, θ̇=0, ψ̇=0]"""
        return torch.zeros((6,))
    
    @property
    def u_equilibrium(self):
        """平衡点控制输入: [τx=0, τy=0, τz=0]"""
        return torch.zeros((3,))


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试 3D 四旋翼姿态动力学")
    print("=" * 60)
    
    # 创建姿态动力学系统
    dynamics = AttitudeDynamics(
        length=0.25, 
        mass=0.486, 
        Ixx=0.00383, 
        Iyy=0.00383, 
        Izz=0.00765
    )
    
    print(f"\n姿态系统参数:")
    print(f"  状态维度: {dynamics.nx}")
    print(f"  控制维度: {dynamics.nu}")
    print(f"  机臂长度: {dynamics.length} m")
    print(f"  质量: {dynamics.mass} kg")
    print(f"  转动惯量: Ixx={dynamics.Ixx}, Iyy={dynamics.Iyy}, Izz={dynamics.Izz} kg·m²")
    print(f"  平衡点状态: {dynamics.x_equilibrium}")
    print(f"  平衡点控制: {dynamics.u_equilibrium}")
    
    # 测试动力学计算
    print("\n" + "=" * 60)
    print("测试动力学前向传播")
    print("=" * 60)
    
    batch_size = 5
    x_test = torch.randn(batch_size, 6) * 0.1  # 小角度扰动
    u_test = torch.randn(batch_size, 3) * 0.01  # 小力矩
    
    print(f"\n测试输入状态 shape: {x_test.shape}")
    print(f"测试输入控制 shape: {u_test.shape}")
    
    x_dot = dynamics.forward(x_test, u_test)
    
    print(f"\n输出状态导数 shape: {x_dot.shape}")
    print(f"状态导数范围: [{x_dot.min().item():.3f}, {x_dot.max().item():.3f}]")
    
    # 测试线性化
    print("\n" + "=" * 60)
    print("测试线性化动力学")
    print("=" * 60)
    
    A, B = dynamics.linearized_dynamics(dynamics.x_equilibrium, dynamics.u_equilibrium)
    
    print(f"\nA 矩阵 shape: {A.shape}")
    print(f"A 矩阵:\n{A}")
    
    print(f"\nB 矩阵 shape: {B.shape}")
    print(f"B 矩阵:\n{B}")
    
    # 测试 LQR
    print("\n" + "=" * 60)
    print("测试 LQR 控制器设计")
    print("=" * 60)
    
    # Q 矩阵：对角线权重
    # 姿态角权重更高，角速度权重稍低
    Q = np.diag([10.0, 10.0, 5.0, 1.0, 1.0, 0.5])
    R = np.diag([1.0, 1.0, 1.0])
    
    K, S = dynamics.lqr_control(Q, R)
    
    print(f"\nLQR 增益矩阵 K shape: {K.shape}")
    print(f"LQR 增益 K:\n{K}")
    
    # 检查闭环稳定性
    A_cl = A + B @ K
    eigenvalues = np.linalg.eigvals(A_cl)
    
    print(f"\n闭环系统特征值:")
    for i, eig in enumerate(eigenvalues):
        print(f"  λ{i+1} = {eig:.4f}")
    
    if np.all(np.real(eigenvalues) < 0):
        print("\n✓ 闭环系统稳定!")
    else:
        print("\n✗ 闭环系统不稳定!")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
