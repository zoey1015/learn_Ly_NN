"""
ROA 采样与姿态解算脚本

功能：
1. 加载训练好的 X/Y/Z 轴控制器和 Lyapunov 网络
2. 从各轴 lyaloss 中恢复 ρ，确定 ROA = {x | V(x) ≤ ρ}
3. 在 ROA 内采样状态，计算控制输出 ux, uy, uz
4. 通过 input_mixer 解算 U1, φ_d, θ_d
5. 统计姿态环的训练范围（limit）

使用方法：
    python roa_sample_and_resolve.py \
        --x_dir output/quadrotor_pos_x/.../final \
        --y_dir output/quadrotor_pos_y/.../final \
        --z_dir output/quadrotor_pos_z/.../final \
        --output_dir ./attitude_limit
"""
import os
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov

from quadrotor3d_dynamic.pos import Position_x_dynamic, PositionYDynamics, PositionZDynamics
from quadrotor3d_dynamic.input_mixer import Attitude_mixer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ============== 配置 ==============

# 各轴独立配置（与训练时保持一致）
AXIS_CFG = {
    'x': {
        'controller_nlayer': 3,
        'controller_hidden_dim': 16,
        'limit': [2.0, 3.0],  # 位置 ±2m, 速度 ±3m/s
        'u_lo': [-20.0],
        'u_up': [20.0],
    },
    'y': {
        'controller_nlayer': 3,
        'controller_hidden_dim': 16,
        'limit': [2.0, 3.0],  # 位置 ±2m, 速度 ±3m/s
        'u_lo': [-20.0],
        'u_up': [20.0],
    },
    'z': {
        'controller_nlayer': 2,
        'controller_hidden_dim': 8,
        'limit': [2.0, 2.5],  # 位置 ±2m, 速度 ±2.5m/s
        'u_lo': [-10.0],
        'u_up': [30.0],
    },
}

# 公共配置
DEFAULT_CFG = {
    'mass': 0.486,
    'gravity': 9.81,
    'dt': 0.01,
    'lyapunov_hidden_widths': [16, 16, 8],
    'R_rows': 2,
    'V_psd_form': 'L1',
    'position_integration': 'ExplicitEuler',
    'velocity_integration': 'ExplicitEuler',
    'rho_multiplier': 1.0,
    'kappa': 0.01,
}

AXIS_CLASS = {
    'x': Position_x_dynamic,
    'y': PositionYDynamics,
    'z': PositionZDynamics,
}


# ============== 加载单轴模型 ==============

def load_axis_model(axis: str, final_dir: str, cfg: dict):
    """
    加载单个轴的控制器、Lyapunov 网络和 lyaloss（含 ρ 信息）。

    Args:
        axis: 'x', 'y', 'z'
        final_dir: 训练输出的 final 目录
        cfg: 公共配置

    Returns:
        controller, lyapunov_nn, lyaloss, dynamics_continuous, axis_cfg
    """
    # 获取该轴的独立配置
    axis_cfg = AXIS_CFG[axis]
    
    dyn_cls = AXIS_CLASS[axis]
    dynamics_continuous = dyn_cls(mass=cfg['mass'], gravity=cfg['gravity'])

    # 离散化（lyaloss 内部会用到）
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(
        dynamics_continuous,
        dt=cfg['dt'],
        position_integration=dynamical_system.IntegrationMethod[cfg['position_integration']],
        velocity_integration=dynamical_system.IntegrationMethod[cfg['velocity_integration']],
    )

    # 控制器（使用各轴独立配置）
    ctrl = controllers.NeuralNetworkController(
        nlayer=axis_cfg['controller_nlayer'],
        in_dim=dynamics_continuous.nx,
        out_dim=dynamics_continuous.nu,
        hidden_dim=axis_cfg['controller_hidden_dim'],
        clip_output="clamp",
        u_lo=torch.tensor(axis_cfg['u_lo'], dtype=dtype),
        u_up=torch.tensor(axis_cfg['u_up'], dtype=dtype),
        x_equilibrium=dynamics_continuous.x_equilibrium,
        u_equilibrium=dynamics_continuous.u_equilibrium,
    )

    # Lyapunov 网络
    lyap = lyapunov.NeuralNetworkLyapunov(
        goal_state=dynamics_continuous.x_equilibrium,
        hidden_widths=cfg['lyapunov_hidden_widths'],
        x_dim=dynamics_continuous.nx,
        R_rows=cfg['R_rows'],
        absolute_output=True,
        eps=0.01,
        activation=nn.LeakyReLU,
        V_psd_form=cfg['V_psd_form'],
    )

    # 加载权重
    ctrl_path = os.path.join(final_dir, "controller_final.pth")
    lyap_path = os.path.join(final_dir, "lyapunov_final.pth")
    lyaloss_path = os.path.join(final_dir, "lyaloss_final.pth")

    ctrl.load_state_dict(torch.load(ctrl_path, map_location=device))
    lyap.load_state_dict(torch.load(lyap_path, map_location=device))

    # 构建 lyaloss 并加载（用于恢复 x_boundary → 算 ρ）
    limit = torch.tensor(axis_cfg['limit'], device=device, dtype=dtype)
    lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics, ctrl, lyap,
        box_lo=-limit,
        box_up=limit,
        rho_multiplier=cfg['rho_multiplier'],
        kappa=cfg['kappa'],
        hard_max=True,
    )
    lyaloss.load_state_dict(torch.load(lyaloss_path, map_location=device))

    # 移到设备
    ctrl.to(device).eval()
    lyap.to(device).eval()
    lyaloss.to(device).eval()

    logger.info(f"[{axis.upper()}] 模型加载完成: {final_dir}")
    logger.info(f"[{axis.upper()}] 网络结构: nlayer={axis_cfg['controller_nlayer']}, hidden_dim={axis_cfg['controller_hidden_dim']}")
    logger.info(f"[{axis.upper()}] limit: {axis_cfg['limit']}, u_range: [{axis_cfg['u_lo']}, {axis_cfg['u_up']}]")

    return ctrl, lyap, lyaloss, dynamics_continuous, axis_cfg


# ============== 提取 ROA ==============

def extract_roa_samples(
    axis: str,
    controller,
    lyapunov_nn,
    lyaloss,
    dynamics_continuous,
    axis_cfg: dict,
    grid_resolution: int = 200,
):
    """
    在训练区域内网格采样，筛选 ROA 内的点，并计算控制输出。

    Args:
        axis: 轴名
        controller: 控制器
        lyapunov_nn: Lyapunov 网络
        lyaloss: LyapunovDerivativeLoss（含 ρ）
        dynamics_continuous: 连续时间动力学
        axis_cfg: 该轴的配置（含 limit）
        grid_resolution: 网格分辨率

    Returns:
        roa_states: ROA 内的状态 [N, 2]
        roa_controls: 对应的控制输出 [N, 1]
        rho: ρ 值
        all_states: 全部网格状态（画图用）
        all_V: 全部 V 值（画图用）
    """
    limit = axis_cfg['limit']

    # 生成网格
    pos_ticks = torch.linspace(-limit[0], limit[0], grid_resolution, device=device, dtype=dtype)
    vel_ticks = torch.linspace(-limit[1], limit[1], grid_resolution, device=device, dtype=dtype)
    grid_pos, grid_vel = torch.meshgrid(pos_ticks, vel_ticks, indexing='ij')
    all_states = torch.stack([grid_pos.flatten(), grid_vel.flatten()], dim=1)

    # 计算 V 和 ρ
    with torch.no_grad():
        all_V = lyapunov_nn(all_states).squeeze()
        rho = lyaloss.get_rho().item()

    # 筛选 ROA 内的点
    roa_mask = all_V <= rho
    roa_states = all_states[roa_mask]

    # 计算控制输出
    with torch.no_grad():
        roa_controls = controller(roa_states)

    roa_ratio = roa_mask.float().mean().item() * 100

    logger.info(f"[{axis.upper()}] ρ = {rho:.4f}")
    logger.info(f"[{axis.upper()}] ROA 点数: {roa_states.shape[0]} / {all_states.shape[0]} ({roa_ratio:.1f}%)")
    logger.info(f"[{axis.upper()}] 控制输出范围: [{roa_controls.min().item():.4f}, {roa_controls.max().item():.4f}]")

    # 也记录 ROA 内的位置/速度范围
    if roa_states.shape[0] > 0:
        logger.info(f"[{axis.upper()}] ROA 位置范围: [{roa_states[:,0].min().item():.4f}, {roa_states[:,0].max().item():.4f}]")
        logger.info(f"[{axis.upper()}] ROA 速度范围: [{roa_states[:,1].min().item():.4f}, {roa_states[:,1].max().item():.4f}]")

    return roa_states, roa_controls, rho, all_states, all_V


# ============== 组合三轴 + 解算 ==============

def combine_and_resolve(
    roa_x_states, roa_x_controls,
    roa_y_states, roa_y_controls,
    roa_z_states, roa_z_controls,
    cfg: dict,
    num_samples: int = 50000,
    psi_ref: float = 0.0,
):
    """
    从三轴 ROA 中采样，组合 (ux, uy, uz)，解算姿态命令。

    策略：从各轴 ROA 控制输出中独立随机采样，组成 (ux, uy, uz) 的组合。

    Args:
        roa_*_states: 各轴 ROA 内的状态
        roa_*_controls: 各轴 ROA 内的控制输出
        cfg: 配置
        num_samples: 采样数量
        psi_ref: 偏航角参考值

    Returns:
        att_cmd: 期望姿态 [N, 3]
        U1: 总推力 [N, 1]
        u_pos: 三轴加速度 [N, 3]
        sampled_states: 采样的状态 dict
    """
    mixer = Attitude_mixer(mass=cfg['mass'], gravity=cfg['gravity'])

    n_x = roa_x_controls.shape[0]
    n_y = roa_y_controls.shape[0]
    n_z = roa_z_controls.shape[0]

    logger.info(f"\n各轴 ROA 可用点数: X={n_x}, Y={n_y}, Z={n_z}")

    # 独立采样索引
    idx_x = torch.randint(0, n_x, (num_samples,), device=device)
    idx_y = torch.randint(0, n_y, (num_samples,), device=device)
    idx_z = torch.randint(0, n_z, (num_samples,), device=device)

    ux = roa_x_controls[idx_x]  # [N, 1]
    uy = roa_y_controls[idx_y]  # [N, 1]
    uz = roa_z_controls[idx_z]  # [N, 1]

    u_pos = torch.cat([ux, uy, uz], dim=1)  # [N, 3]
    psi = torch.full((num_samples, 1), psi_ref, device=device, dtype=dtype)

    # 解算：uz 来自 PositionZDynamics 控制器，已包含重力补偿
    att_cmd, U1 = mixer.compute(u_pos, psi, gravity_compensated=True)

    # 记录采样的状态（供分析用）
    sampled_states = {
        'x_states': roa_x_states[idx_x],
        'y_states': roa_y_states[idx_y],
        'z_states': roa_z_states[idx_z],
    }

    return att_cmd, U1, u_pos, sampled_states


# ============== 统计与输出 ==============

def compute_statistics(att_cmd, U1, u_pos, cfg):
    """
    统计解算结果，给出姿态环的训练范围建议。
    """
    phi = att_cmd[:, 0]
    theta = att_cmd[:, 1]

    results = {
        'ux_range': [u_pos[:, 0].min().item(), u_pos[:, 0].max().item()],
        'uy_range': [u_pos[:, 1].min().item(), u_pos[:, 1].max().item()],
        'uz_range': [u_pos[:, 2].min().item(), u_pos[:, 2].max().item()],
        'phi_range': [phi.min().item(), phi.max().item()],
        'theta_range': [theta.min().item(), theta.max().item()],
        'phi_abs_max': phi.abs().max().item(),
        'theta_abs_max': theta.abs().max().item(),
        'U1_range': [U1.min().item(), U1.max().item()],
        'U1_hover': cfg['mass'] * cfg['gravity'],
    }

    logger.info("\n" + "=" * 70)
    logger.info("解算结果统计")
    logger.info("=" * 70)

    logger.info(f"\n--- 位置环控制输出范围 ---")
    logger.info(f"  ux ∈ [{results['ux_range'][0]:.4f}, {results['ux_range'][1]:.4f}] m/s²")
    logger.info(f"  uy ∈ [{results['uy_range'][0]:.4f}, {results['uy_range'][1]:.4f}] m/s²")
    logger.info(f"  uz ∈ [{results['uz_range'][0]:.4f}, {results['uz_range'][1]:.4f}] m/s² (含重力补偿)")

    logger.info(f"\n--- 解算后的姿态范围 ---")
    logger.info(f"  φ (roll)  ∈ [{np.degrees(results['phi_range'][0]):.2f}°, {np.degrees(results['phi_range'][1]):.2f}°]")
    logger.info(f"  θ (pitch) ∈ [{np.degrees(results['theta_range'][0]):.2f}°, {np.degrees(results['theta_range'][1]):.2f}°]")
    logger.info(f"  |φ|_max = {np.degrees(results['phi_abs_max']):.2f}°")
    logger.info(f"  |θ|_max = {np.degrees(results['theta_abs_max']):.2f}°")

    logger.info(f"\n--- 推力范围 ---")
    logger.info(f"  U1 ∈ [{results['U1_range'][0]:.4f}, {results['U1_range'][1]:.4f}] N")
    logger.info(f"  U1_hover = {results['U1_hover']:.4f} N")

    # 给出建议的姿态环 limit（加 10% 裕度）
    margin = 1.1
    phi_limit = results['phi_abs_max'] * margin
    theta_limit = results['theta_abs_max'] * margin

    logger.info(f"\n--- 姿态环训练建议 ---")
    logger.info(f"  姿态角 limit (加10%裕度):")
    logger.info(f"    φ_limit = {phi_limit:.4f} rad ({np.degrees(phi_limit):.2f}°)")
    logger.info(f"    θ_limit = {theta_limit:.4f} rad ({np.degrees(theta_limit):.2f}°)")
    logger.info(f"  姿态角速度 limit 建议: 取 2~3 倍角度 limit")
    logger.info(f"    ω_φ_limit ≈ {2 * phi_limit:.4f} rad/s")
    logger.info(f"    ω_θ_limit ≈ {2 * theta_limit:.4f} rad/s")
    logger.info(f"  姿态环平衡点:")
    logger.info(f"    φ* = 0, θ* = 0, ψ* = 0")
    logger.info(f"    ω* = [0, 0, 0]")
    logger.info(f"    U1* = {results['U1_hover']:.4f} N (= mg)")

    results['phi_limit_suggested'] = phi_limit
    results['theta_limit_suggested'] = theta_limit

    return results


# ============== 绘图 ==============

def plot_results(
    roa_x_states, roa_x_controls, all_x_states, all_x_V, rho_x, limit_x,
    roa_y_states, roa_y_controls, all_y_states, all_y_V, rho_y, limit_y,
    roa_z_states, roa_z_controls, all_z_states, all_z_V, rho_z, limit_z,
    att_cmd, U1, u_pos,
    cfg, save_dir,
):
    """绘制 ROA + 解算结果图"""
    grid_res = int(np.sqrt(all_x_states.shape[0]))
    limits = {'X': limit_x, 'Y': limit_y, 'Z': limit_z}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # ========== 上排：三轴 ROA 热力图 ==========
    for i, (axis, all_st, all_v, rho, roa_st, roa_ctrl) in enumerate([
        ('X', all_x_states, all_x_V, rho_x, roa_x_states, roa_x_controls),
        ('Y', all_y_states, all_y_V, rho_y, roa_y_states, roa_y_controls),
        ('Z', all_z_states, all_z_V, rho_z, roa_z_states, roa_z_controls),
    ]):
        ax = axes[0, i]
        limit = limits[axis]  # 使用各轴独立的 limit
        V_grid = all_v.cpu().numpy().reshape(grid_res, grid_res)
        pos_ticks = np.linspace(-limit[0], limit[0], grid_res)
        vel_ticks = np.linspace(-limit[1], limit[1], grid_res)

        # V 热力图
        im = ax.pcolormesh(
            pos_ticks, vel_ticks, V_grid.T,
            shading='auto', cmap='viridis',
            vmax=rho * 3,  # 限制颜色范围
        )
        # ROA 边界（V = ρ 等值线）
        ax.contour(
            pos_ticks, vel_ticks, V_grid.T,
            levels=[rho], colors='red', linewidths=2,
        )
        ax.set_xlabel(f'{axis} position (m)')
        ax.set_ylabel(f'{axis} velocity (m/s)')
        ax.set_title(f'{axis}-axis: V(x) & ROA (ρ={rho:.2f})')
        plt.colorbar(im, ax=ax, label='V(x)')

    # ========== 下排：解算结果分布 ==========
    phi = att_cmd[:, 0].cpu().numpy()
    theta = att_cmd[:, 1].cpu().numpy()
    U1_np = U1.squeeze().cpu().numpy()

    # φ 分布
    ax = axes[1, 0]
    ax.hist(np.degrees(phi), bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(0, color='red', linestyle='--', label='平衡点 φ=0')
    ax.set_xlabel('φ_d (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Roll (φ) distribution')
    ax.legend()

    # θ 分布
    ax = axes[1, 1]
    ax.hist(np.degrees(theta), bins=100, color='coral', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(0, color='red', linestyle='--', label='平衡点 θ=0')
    ax.set_xlabel('θ_d (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Pitch (θ) distribution')
    ax.legend()

    # U1 分布
    ax = axes[1, 2]
    ax.hist(U1_np, bins=100, color='mediumseagreen', alpha=0.7, edgecolor='black', linewidth=0.3)
    U1_hover = cfg['mass'] * cfg['gravity']
    ax.axvline(U1_hover, color='red', linestyle='--', label=f'悬停 U1={U1_hover:.2f}N')
    ax.set_xlabel('U1 (N)')
    ax.set_ylabel('Count')
    ax.set_title('Thrust (U1) distribution')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'roa_sample_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"\n图片已保存: {save_path}")


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description='ROA 采样与姿态解算')
    parser.add_argument('--x_dir', type=str, required=True, help='X轴 final 目录路径')
    parser.add_argument('--y_dir', type=str, required=True, help='Y轴 final 目录路径')
    parser.add_argument('--z_dir', type=str, required=True, help='Z轴 final 目录路径')
    parser.add_argument('--output_dir', type=str, default='./attitude_limit', help='输出目录')
    parser.add_argument('--grid_resolution', type=int, default=200, help='ROA 采样网格分辨率')
    parser.add_argument('--num_samples', type=int, default=50000, help='组合采样数量')
    parser.add_argument('--psi_ref', type=float, default=0.0, help='偏航角参考值 (rad)')
    parser.add_argument('--no_plot', action='store_true', help='不绘图')
    args = parser.parse_args()

    cfg = DEFAULT_CFG
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ROA 采样与姿态解算")
    logger.info("=" * 70)

    # ---------- 1. 加载三轴模型 ----------
    logger.info("\n--- 加载模型 ---")
    ctrl_x, lyap_x, lyaloss_x, dyn_x, cfg_x = load_axis_model('x', args.x_dir, cfg)
    ctrl_y, lyap_y, lyaloss_y, dyn_y, cfg_y = load_axis_model('y', args.y_dir, cfg)
    ctrl_z, lyap_z, lyaloss_z, dyn_z, cfg_z = load_axis_model('z', args.z_dir, cfg)

    # ---------- 2. 提取 ROA ----------
    logger.info("\n--- 提取 ROA ---")
    roa_x_st, roa_x_ctrl, rho_x, all_x_st, all_x_V = extract_roa_samples(
        'x', ctrl_x, lyap_x, lyaloss_x, dyn_x, cfg_x, args.grid_resolution)
    roa_y_st, roa_y_ctrl, rho_y, all_y_st, all_y_V = extract_roa_samples(
        'y', ctrl_y, lyap_y, lyaloss_y, dyn_y, cfg_y, args.grid_resolution)
    roa_z_st, roa_z_ctrl, rho_z, all_z_st, all_z_V = extract_roa_samples(
        'z', ctrl_z, lyap_z, lyaloss_z, dyn_z, cfg_z, args.grid_resolution)

    # ---------- 3. 组合采样 + 解算 ----------
    logger.info("\n--- 组合采样与解算 ---")
    att_cmd, U1, u_pos, sampled_states = combine_and_resolve(
        roa_x_st, roa_x_ctrl,
        roa_y_st, roa_y_ctrl,
        roa_z_st, roa_z_ctrl,
        cfg,
        num_samples=args.num_samples,
        psi_ref=args.psi_ref,
    )

    # ---------- 4. 统计结果 ----------
    results = compute_statistics(att_cmd, U1, u_pos, cfg)

    # ---------- 5. 保存结果 ----------
    # 保存数值结果
    torch.save({
        'results': results,
        'att_cmd': att_cmd.cpu(),
        'U1': U1.cpu(),
        'u_pos': u_pos.cpu(),
        'rho_x': rho_x, 'rho_y': rho_y, 'rho_z': rho_z,
        'cfg': cfg,
    }, os.path.join(output_dir, 'attitude_limit_data.pt'))

    # 保存建议的 limit 为简洁格式
    with open(os.path.join(output_dir, 'attitude_limit_suggestion.txt'), 'w') as f:
        f.write("姿态环训练范围建议 (由位置环 ROA 推导)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"φ_limit = {results['phi_limit_suggested']:.6f} rad "
                f"({np.degrees(results['phi_limit_suggested']):.2f}°)\n")
        f.write(f"θ_limit = {results['theta_limit_suggested']:.6f} rad "
                f"({np.degrees(results['theta_limit_suggested']):.2f}°)\n")
        f.write(f"\nφ_range = [{np.degrees(results['phi_range'][0]):.2f}°, "
                f"{np.degrees(results['phi_range'][1]):.2f}°]\n")
        f.write(f"θ_range = [{np.degrees(results['theta_range'][0]):.2f}°, "
                f"{np.degrees(results['theta_range'][1]):.2f}°]\n")
        f.write(f"\nU1_range = [{results['U1_range'][0]:.4f}, {results['U1_range'][1]:.4f}] N\n")
        f.write(f"U1_hover = {results['U1_hover']:.4f} N\n")
        f.write(f"\n姿态角速度 limit 建议:\n")
        f.write(f"  ω_φ ≈ {2 * results['phi_limit_suggested']:.4f} rad/s\n")
        f.write(f"  ω_θ ≈ {2 * results['theta_limit_suggested']:.4f} rad/s\n")
        f.write(f"\nρ 值: X={rho_x:.4f}, Y={rho_y:.4f}, Z={rho_z:.4f}\n")
        f.write(f"采样数: {args.num_samples}\n")
        f.write(f"网格分辨率: {args.grid_resolution}\n")

    logger.info(f"\n结果已保存到: {output_dir}")

    # ---------- 6. 绘图 ----------
    if not args.no_plot:
        plot_results(
            roa_x_st, roa_x_ctrl, all_x_st, all_x_V, rho_x, cfg_x['limit'],
            roa_y_st, roa_y_ctrl, all_y_st, all_y_V, rho_y, cfg_y['limit'],
            roa_z_st, roa_z_ctrl, all_z_st, all_z_V, rho_z, cfg_z['limit'],
            att_cmd, U1, u_pos,
            cfg, output_dir,
        )


if __name__ == "__main__":
    main()

