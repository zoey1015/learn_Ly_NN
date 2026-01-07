"""
Nonlinear State-Dependent Function (NSDF) Transformation
Based on Song et al. (2019) for state constraints
"""

import torch
import torch.nn as nn
from typing import Tuple
import math


class NSDFTransform:
    """
    NSDF transformation for position constraints.
    
    For 2D quadrotor: only transform x and y positions
    Other states (θ, velocities) remain unchanged
    """
    
    def __init__(
        self,
        Fx1: float = 1.0,
        Fx2: float = 1.0,
        Fy1: float = 0.5,
        Fy2: float = 1.5,
        epsilon: float = 1e-6,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            Fx1, Fx2: x position bounds (-Fx1, Fx2)
            Fy1, Fy2: y position bounds (-Fy1, Fy2)
            epsilon: Numerical stability constant
        """
        self.Fx1 = Fx1
        self.Fx2 = Fx2
        self.Fy1 = Fy1
        self.Fy2 = Fy2
        self.epsilon = epsilon
        self.device = device
        self.dtype = dtype
        
        print(f"NSDF Transform initialized: x ∈ (-{Fx1}, {Fx2}), y ∈ (-{Fy1}, {Fy2})")
    
    def forward_transform(self, x_quad: torch.Tensor) -> torch.Tensor:
        """
        Transform physical state to virtual state.
        
        x_quad = [x, y, θ, ẋ, ẏ, θ̇] → ξ = [ζ_x, ζ_y, θ, ẋ, ẏ, θ̇]
        
        Args:
            x_quad: Physical state [..., 6]
        Returns:
            xi: Virtual state [..., 6]
        """
        # Extract position
        x = x_quad[..., 0]
        y = x_quad[..., 1]
        
        # Clamp to safe region
        x_clamped = torch.clamp(x, 
                               min=-self.Fx1 + self.epsilon,
                               max=self.Fx2 - self.epsilon)
        y_clamped = torch.clamp(y,
                               min=-self.Fy1 + self.epsilon,
                               max=self.Fy2 - self.epsilon)
        
        # Compute NSDF
        denom_x = (self.Fx1 + x_clamped) * (self.Fx2 - x_clamped)
        denom_y = (self.Fy1 + y_clamped) * (self.Fy2 - y_clamped)
        
        denom_x = torch.clamp(denom_x, min=self.epsilon)
        denom_y = torch.clamp(denom_y, min=self.epsilon)
        
        zeta_x = x_clamped / denom_x
        zeta_y = y_clamped / denom_y
        
        # Construct virtual state (only replace x, y)
        xi = x_quad.clone()
        xi[..., 0] = zeta_x
        xi[..., 1] = zeta_y
        
        return xi
    
    def inverse_transform(self, xi: torch.Tensor, max_iter: int = 20) -> torch.Tensor:
        """
        Transform virtual state back to physical state.
        
        ξ = [ζ_x, ζ_y, θ, ẋ, ẏ, θ̇] → x_quad = [x, y, θ, ẋ, ẏ, θ̇]
        
        Solve: ζ_x = x / ((Fx1 + x)(Fx2 - x)) for x
        
        Args:
            xi: Virtual state [..., 6]
            max_iter: Maximum Newton-Raphson iterations
        Returns:
            x_quad: Physical state [..., 6]
        """
        zeta_x = xi[..., 0]
        zeta_y = xi[..., 1]
        
        # Solve for x and y using Newton-Raphson
        x = self._solve_cubic(zeta_x, self.Fx1, self.Fx2, max_iter)
        y = self._solve_cubic(zeta_y, self.Fy1, self.Fy2, max_iter)
        
        # Construct physical state
        x_quad = xi.clone()
        x_quad[..., 0] = x
        x_quad[..., 1] = y
        
        return x_quad
    
    def _solve_cubic(
        self, 
        zeta: torch.Tensor, 
        F1: float, 
        F2: float,
        max_iter: int
    ) -> torch.Tensor:
        """
        Solve cubic equation: ζ*(F1 + x)*(F2 - x) - x = 0
        Using Newton-Raphson method.
        
        Args:
            zeta: NSDF value
            F1, F2: Constraint bounds
            max_iter: Maximum iterations
        Returns:
            x: Position value
        """
        # Initial guess: linear approximation near origin
        x = zeta * F1 * F2
        
        for _ in range(max_iter):
            # f(x) = ζ*(F1 + x)*(F2 - x) - x
            f = zeta * (F1 + x) * (F2 - x) - x
            
            # f'(x) = ζ*(F2 - x - F1 - x) - 1 = ζ*(F2 - F1 - 2x) - 1
            df = zeta * (F2 - F1 - 2*x) - 1
            
            # Newton-Raphson update
            dx = -f / (df + self.epsilon)
            x = x + dx
            
            # Check convergence
            if torch.abs(dx).max() < 1e-6:
                break
        
        # Clamp to constraint bounds
        x = torch.clamp(x, min=-F1 + self.epsilon, max=F2 - self.epsilon)
        
        return x
    
    def compute_mu(self, x_quad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute μ function (needed for Song's control law).
        
        μ_i = (Fi1*Fi2 + xi²) / ((Fi1 + xi)² * (Fi2 - xi)²)
        """
        x = x_quad[..., 0]
        y = x_quad[..., 1]
        
        x_clamped = torch.clamp(x, -self.Fx1 + self.epsilon, self.Fx2 - self.epsilon)
        y_clamped = torch.clamp(y, -self.Fy1 + self.epsilon, self.Fy2 - self.epsilon)
        
        num_x = self.Fx1 * self.Fx2 + x_clamped ** 2
        num_y = self.Fy1 * self.Fy2 + y_clamped ** 2
        
        denom_x = ((self.Fx1 + x_clamped) ** 2) * ((self.Fx2 - x_clamped) ** 2)
        denom_y = ((self.Fy1 + y_clamped) ** 2) * ((self.Fy2 - y_clamped) ** 2)
        
        denom_x = torch.clamp(denom_x, min=self.epsilon)
        denom_y = torch.clamp(denom_y, min=self.epsilon)
        
        mu_x = num_x / denom_x
        mu_y = num_y / denom_y
        
        return mu_x, mu_y
    
    def compute_dmu_dx(self, x_quad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 dμ/dx 用于速度耦合项
    
        μ = (F1*F2 + x²) / [(F1 + x)²(F2 - x)²]
    
        dμ/dx = [2x·(F1+x)²(F2-x)² - (F1*F2+x²)·d/dx[(F1+x)²(F2-x)²]] / [(F1+x)²(F2-x)²]²
    
        其中: d/dx[(F1+x)²(F2-x)²] = 2(F1+x)(F2-x)² - 2(F1+x)²(F2-x)
        """
        x = x_quad[..., 0]
        y = x_quad[..., 1]
    
        x_clamped = torch.clamp(x, -self.Fx1 + self.epsilon, self.Fx2 - self.epsilon)
        y_clamped = torch.clamp(y, -self.Fy1 + self.epsilon, self.Fy2 - self.epsilon)
    
        # 对 x 方向
        p1_x = self.Fx1 + x_clamped
        p2_x = self.Fx2 - x_clamped
        num_x = self.Fx1 * self.Fx2 + x_clamped ** 2
        denom_x = (p1_x ** 2) * (p2_x ** 2)
    
        # dμ/dx 的分子
        term1_x = 2 * x_clamped * denom_x
        term2_x = num_x * (2 * p1_x * (p2_x ** 2) - 2 * (p1_x ** 2) * p2_x)
        dmu_dx = (term1_x - term2_x) / (denom_x ** 2 + self.epsilon)
    
        # 对 y 方向（相同逻辑）
        p1_y = self.Fy1 + y_clamped
        p2_y = self.Fy2 - y_clamped
        num_y = self.Fy1 * self.Fy2 + y_clamped ** 2
        denom_y = (p1_y ** 2) * (p2_y ** 2)
    
        term1_y = 2 * y_clamped * denom_y
        term2_y = num_y * (2 * p1_y * (p2_y ** 2) - 2 * (p1_y ** 2) * p2_y)
        dmu_dy = (term1_y - term2_y) / (denom_y ** 2 + self.epsilon)
    
        return dmu_dx, dmu_dy


def test_nsdf_transform():
    """Test NSDF forward and inverse transforms."""
    print("="*60)
    print("Testing NSDF Transform")
    print("="*60)
    
    transform = NSDFTransform()
    
    # Test case 1: Origin
    x_quad = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    xi = transform.forward_transform(x_quad)
    x_recovered = transform.inverse_transform(xi)
    
    print(f"\nTest 1: Origin")
    print(f"  Original:  {x_quad[0, :2]}")
    print(f"  ζ:         {xi[0, :2]}")
    print(f"  Recovered: {x_recovered[0, :2]}")
    print(f"  Error:     {torch.abs(x_quad - x_recovered).max().item():.2e}")
    
    # Test case 2: Random positions
    x_quad = torch.tensor([[0.5, 0.8, 0.1, 0.0, 0.0, 0.0]])
    xi = transform.forward_transform(x_quad)
    x_recovered = transform.inverse_transform(xi)
    
    print(f"\nTest 2: Random position")
    print(f"  Original:  {x_quad[0, :2]}")
    print(f"  ζ:         {xi[0, :2]}")
    print(f"  Recovered: {x_recovered[0, :2]}")
    print(f"  Error:     {torch.abs(x_quad - x_recovered).max().item():.2e}")
    
    # Test case 3: Batch
    batch_size = 100
    x_batch = torch.randn(batch_size, 6) * 0.5
    x_batch[:, 0] = torch.clamp(x_batch[:, 0], -0.9, 0.9)
    x_batch[:, 1] = torch.clamp(x_batch[:, 1], -0.45, 1.45)
    
    xi_batch = transform.forward_transform(x_batch)
    x_recovered_batch = transform.inverse_transform(xi_batch)
    
    max_error = torch.abs(x_batch - x_recovered_batch).max().item()
    mean_error = torch.abs(x_batch - x_recovered_batch).mean().item()
    
    print(f"\nTest 3: Batch ({batch_size} samples)")
    print(f"  Max error:  {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    
    if max_error < 1e-5:
        print("\n✓ All tests passed!")
        return True
    else:
        print("\n✗ Tests failed!")
        return False

class TanTransform:
    def __init__(self, Lx=1.0, Ly=1.0, epsilon=1e-6, zeta_max=20.0, device=None, dtype=torch.float32):
        assert Lx > 0 and Ly > 0, "Lx and Ly must be positive."
        self.Lx, self.Ly = Lx, Ly
        self.eps = epsilon
        self.zeta_max = zeta_max
        self.device = device
        self.dtype = dtype

        # k = pi/(2L)
        self.kx = math.pi / (2.0 * Lx)
        self.ky = math.pi / (2.0 * Ly)

    def forward_transform(self, x_phys: torch.Tensor) -> torch.Tensor:
        xi = x_phys.clone()
        x = torch.clamp(x_phys[:, 0], -self.Lx + self.eps, self.Lx - self.eps)
        y = torch.clamp(x_phys[:, 1], -self.Ly + self.eps, self.Ly - self.eps)

        zeta_x = torch.tan(self.kx * x)
        zeta_y = torch.tan(self.ky * y)

        # clamp to avoid huge zeta in training/bounds
        zeta_x = torch.clamp(zeta_x, -self.zeta_max, self.zeta_max)
        zeta_y = torch.clamp(zeta_y, -self.zeta_max, self.zeta_max)

        xi[:, 0] = zeta_x
        xi[:, 1] = zeta_y
        return xi

    def inverse_transform(self, xi: torch.Tensor) -> torch.Tensor:
        x_phys = xi.clone()
        zeta_x = torch.clamp(xi[:, 0], -self.zeta_max, self.zeta_max)
        zeta_y = torch.clamp(xi[:, 1], -self.zeta_max, self.zeta_max)

        # x = atan(zeta) / k   (no epsilon here)
        x = torch.atan(zeta_x) / self.kx
        y = torch.atan(zeta_y) / self.ky

        x = torch.clamp(x, -self.Lx + self.eps, self.Lx - self.eps)
        y = torch.clamp(y, -self.Ly + self.eps, self.Ly - self.eps)

        x_phys[:, 0] = x
        x_phys[:, 1] = y
        return x_phys

    def compute_mu(self, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zeta_x = torch.clamp(xi[:, 0], -self.zeta_max, self.zeta_max)
        zeta_y = torch.clamp(xi[:, 1], -self.zeta_max, self.zeta_max)
        mu_x = self.kx * (1.0 + zeta_x**2)
        mu_y = self.ky * (1.0 + zeta_y**2)
        return mu_x, mu_y

    def compute_dmu_dx(self, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zeta_x = torch.clamp(xi[:, 0], -self.zeta_max, self.zeta_max)
        zeta_y = torch.clamp(xi[:, 1], -self.zeta_max, self.zeta_max)
        dmu_dx = 2.0 * (self.kx**2) * zeta_x * (1.0 + zeta_x**2)
        dmu_dy = 2.0 * (self.ky**2) * zeta_y * (1.0 + zeta_y**2)
        return dmu_dx, dmu_dy


if __name__ == "__main__":
    test_nsdf_transform()
