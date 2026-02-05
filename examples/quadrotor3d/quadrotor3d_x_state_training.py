"""
四旋翼 X 轴位置控制器训练文件
参考 pendulum_state_training.py 结构
"""
import os
import hydra
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import wandb

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.train_utils as train_utils
# 导入你的位置动力学
from quadrotor3d_dynamic.pos import Position_x_dynamic
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


# ============== 核心：用 LQR 近似初始化网络 ==============

def approximate_controller(
    dynamics: Position_x_dynamic,
    controller: controllers.NeuralNetworkController,
    lyapunov_nn: lyapunov.NeuralNetworkLyapunov,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    Q: np.ndarray,
    R: np.ndarray,
    lr: float,
    max_iter: int,
    logger,
):
    """
    用 LQR 控制器近似初始化神经网络控制器和 Lyapunov 网络
    
    这是生成预训练权重的核心函数
    """
    # 使用你 pos.py 中已有的 lqr_control 方法
    K, S = dynamics.lqr_control(Q, R)
    
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)
    
    logger.info(f"LQR 增益 K: {K}")
    logger.info(f"Lyapunov 矩阵 S:\n{S}")
    
    # 验证闭环稳定性
    A, B = dynamics.linearized_dynamics(
        dynamics.x_equilibrium.numpy(), 
        dynamics.u_equilibrium.numpy()
    )
    A_cl = A + B @ K
    eigvals = np.linalg.eigvals(A_cl)
    logger.info(f"闭环极点: {eigvals}")
    
    # ========== 生成训练数据 ==========
    num_samples = 100000
    x = torch.rand((num_samples, dynamics.nx), dtype=dtype, device=device)
    # 映射到 [lower_limit, upper_limit]
    x = x * (upper_limit - lower_limit) + lower_limit
    
    # 计算 LQR 目标
    # V_target = x^T S x
    V_target = torch.sum(x * (x @ S_torch), dim=1, keepdim=True)
    
    # u_target = K @ x + u_equilibrium
    u_eq = dynamics.u_equilibrium.to(device)
    u_target = x @ K_torch.T + u_eq
    
    # ========== 近似控制器 ==========
    logger.info("=" * 50)
    logger.info("开始近似控制器...")
    
    optimizer_ctrl = torch.optim.Adam(controller.parameters(), lr=lr)
    
    for i in range(max_iter):
        optimizer_ctrl.zero_grad()
        u_pred = controller.forward(x)
        loss_ctrl = torch.nn.MSELoss()(u_pred, u_target)
        
        if i % 100 == 0:
            logger.info(f"[Controller] iter {i}, loss {loss_ctrl.item():.6f}")
        
        loss_ctrl.backward()
        optimizer_ctrl.step()
    
    logger.info(f"[Controller] Final loss: {loss_ctrl.item():.6f}")
    
    # ========== 近似 Lyapunov 函数 ==========
    logger.info("=" * 50)
    logger.info("开始近似 Lyapunov 函数...")
    
    optimizer_lyap = torch.optim.Adam(lyapunov_nn.parameters(), lr=lr)
    
    for i in range(max_iter * 2):  # Lyapunov 通常需要更多迭代
        optimizer_lyap.zero_grad()
        V_pred = lyapunov_nn.forward(x)
        loss_lyap = torch.nn.MSELoss()(V_pred, V_target)
        
        if i % 100 == 0:
            logger.info(f"[Lyapunov] iter {i}, loss {loss_lyap.item():.6f}")
        
        loss_lyap.backward()
        optimizer_lyap.step()
    
    logger.info(f"[Lyapunov] Final loss: {loss_lyap.item():.6f}")
    
    return K, S


def verify_approximation(
    dynamics: Position_x_dynamic,
    controller: controllers.NeuralNetworkController,
    lyapunov_nn,
    K: np.ndarray,
    S: np.ndarray,
    logger,
):
    """验证预训练网络与 LQR 的近似程度"""
    
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)
    u_eq = dynamics.u_equilibrium.to(device)
    
    # 测试点
    test_states = torch.tensor([
        [0.0, 0.0],     # 平衡点
        [1.0, 0.0],     # 位置偏移
        [0.0, 1.0],     # 速度偏移
        [1.0, 1.0],     # 两者偏移
        [-1.0, -1.0],   # 负偏移
        [2.0, 3.0],     # 边界
        [-2.0, -3.0],   # 边界
    ], dtype=dtype, device=device)
    
    logger.info("=" * 60)
    logger.info("预训练验证")
    logger.info("=" * 60)
    logger.info(f"{'State':<20} {'u_NN':>10} {'u_LQR':>10} {'V_NN':>10} {'V_LQR':>10}")
    logger.info("-" * 60)
    
    with torch.no_grad():
        for state in test_states:
            x = state.unsqueeze(0)
            
            # 网络输出
            u_nn = controller.forward(x).item()
            V_nn = lyapunov_nn.forward(x).item()
            
            # LQR 目标
            u_lqr = (x @ K_torch.T + u_eq).item()
            V_lqr = torch.sum(x * (x @ S_torch)).item()
            
            logger.info(f"[{state[0]:>6.2f}, {state[1]:>6.2f}] "
                       f"{u_nn:>10.4f} {u_lqr:>10.4f} {V_nn:>10.4f} {V_lqr:>10.4f}")




# ============== 主训练函数 ==============

@hydra.main(
    config_path="../config",  # config files live in examples/config
    config_name="quadrotor_pos_x_state_training",
    version_base=None,
)
def main(cfg: DictConfig):
    """主训练函数"""
    
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    train_utils.set_seed(cfg.seed)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("四旋翼 X 轴位置控制器训练")
    logger.info("=" * 60)
    
    # ========== 创建动力学系统 ==========
    dt = cfg.model.dt
    position_continuous = Position_x_dynamic(
        mass=cfg.model.mass,
        gravity=cfg.model.gravity,
    )
    
    # 离散化
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(
        position_continuous,
        dt=dt,
        position_integration=dynamical_system.IntegrationMethod[cfg.model.position_integration],
        velocity_integration=dynamical_system.IntegrationMethod[cfg.model.velocity_integration],
    )
    
    # ========== 定义兴趣区域 ==========
    # 位置: ±2m, 速度: ±3m/s
    limit = torch.tensor(cfg.model.limit, device=device, dtype=dtype)  # [2.0, 3.0]
    lower_limit = -limit
    upper_limit = limit
    
    logger.info(f"兴趣区域: 位置 [{-limit[0].item()}, {limit[0].item()}] m")
    logger.info(f"         速度 [{-limit[1].item()}, {limit[1].item()}] m/s")
    
    # ========== 创建控制器网络 ==========
    controller = controllers.NeuralNetworkController(
        nlayer=cfg.model.controller_nlayer,
        in_dim=position_continuous.nx,
        out_dim=position_continuous.nu,
        hidden_dim=cfg.model.controller_hidden_dim,
        clip_output="clamp",
        u_lo=torch.tensor(cfg.model.u_lo, dtype=dtype),
        u_up=torch.tensor(cfg.model.u_up, dtype=dtype),
        x_equilibrium=position_continuous.x_equilibrium,
        u_equilibrium=position_continuous.u_equilibrium,
    )
    controller.eval()
    
    # ========== 创建 Lyapunov 网络 ==========
    lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
        goal_state=position_continuous.x_equilibrium,
        hidden_widths=cfg.model.lyapunov.hidden_widths,
        x_dim=position_continuous.nx,
        R_rows=cfg.model.lyapunov.R_rows,
        absolute_output=True,
        eps=0.01,
        activation=nn.LeakyReLU,
        V_psd_form=cfg.model.V_psd_form,
    )
    lyapunov_nn.eval()
    
    # 移到设备
    dynamics.to(device)
    controller.to(device)
    lyapunov_nn.to(device)
    
    # ========== 预训练：用 LQR 近似初始化 ==========
    if cfg.approximate_lqr:
        logger.info("=" * 60)
        logger.info("开始 LQR 近似初始化...")
        logger.info("=" * 60)
        
        Q = np.diag(cfg.model.lqr.Q)  # 例如 [10.0, 1.0]
        R = np.diag(cfg.model.lqr.R)  # 例如 [1.0]
        
        K, S = approximate_controller(
            position_continuous,
            controller,
            lyapunov_nn,
            lower_limit,
            upper_limit,
            Q=Q,
            R=R,
            lr=cfg.pretrain.lr,
            max_iter=cfg.pretrain.max_iter,
            logger=logger,
        )
        
        # 验证预训练效果
        verify_approximation(position_continuous, controller, lyapunov_nn, K, S, logger)
        
        # 保存预训练权重
        #pretrain_dir = os.path.join(os.getcwd(), "pretrain")
        original_cwd = hydra.utils.get_original_cwd()
        pretrain_dir=os.path.join(
             original_cwd,
             "pretrain",
             cfg.wandb_name,
             datetime.now().strftime("%Y-%m-%d"),
             datetime.now().strftime("%H-%M-%S")
        )
        os.makedirs(pretrain_dir, exist_ok=True)
        
        torch.save(controller.state_dict(), 
                  os.path.join(pretrain_dir, "controller_lqr.pth"))
        torch.save(lyapunov_nn.state_dict(),
                  os.path.join(pretrain_dir, "lyapunov_lqr.pth"))
        
        # 保存 LQR 参数供参考
        np.savez(os.path.join(pretrain_dir, "lqr_params.npz"), K=K, S=S, Q=Q, R=R)
        
        logger.info(f"预训练权重已保存到: {pretrain_dir}")
    
    # ========== 加载预训练权重（如果指定）==========
    if cfg.model.load_controller is not None:
        controller.load_state_dict(torch.load(cfg.model.load_controller))
        logger.info(f"加载控制器权重: {cfg.model.load_controller}")
        
    if cfg.model.load_lyapunov is not None:
        lyapunov_nn.load_state_dict(torch.load(cfg.model.load_lyapunov))
        logger.info(f"加载 Lyapunov 权重: {cfg.model.load_lyapunov}")
    
    # ========== 设置 Lyapunov 损失 ==========
    kappa = cfg.model.kappa
    rho_multiplier = cfg.model.rho_multiplier
    
    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=lower_limit,
        box_up=upper_limit,
        rho_multiplier=rho_multiplier,
        kappa=kappa,
        hard_max=cfg.train.hard_max,
    )
    
    positivity_lyaloss = None  # absolute_output=True 时不需要
    
    # ========== W&B 日志 ==========
    if cfg.train.wandb.enabled:
        wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
            name=cfg.train.wandb.name,
        )
    
    # ========== 主训练循环 ==========
    if not cfg.train.enabled:
        logger.info("训练未启用，仅执行预训练")
        return
        
    grid_size = torch.tensor(cfg.train.grid_size, device=device)
    
    for n, limit_scale in enumerate(cfg.model.limit_scale):
        scaled_limit = limit_scale * limit
        scaled_lower = -scaled_limit
        scaled_upper = scaled_limit
        
        logger.info(f"\n{'='*60}")
        logger.info(f"训练阶段 {n+1}/{len(cfg.model.limit_scale)}, scale={limit_scale}")
        logger.info(f"区域: pos=[{-scaled_limit[0].item():.2f}, {scaled_limit[0].item():.2f}], "
                   f"vel=[{-scaled_limit[1].item():.2f}, {scaled_limit[1].item():.2f}]")
        logger.info("=" * 60)
        
        derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
            dynamics,
            controller,
            lyapunov_nn,
            box_lo=scaled_lower,
            box_up=scaled_upper,
            rho_multiplier=rho_multiplier,
            kappa=kappa,
            hard_max=cfg.train.hard_max,
        )
        
        save_path = os.path.join(os.getcwd(), f"lyaloss_scale{limit_scale}.pth")
        
        candidate_roa_states = limit_scale * torch.tensor(
            cfg.loss.candidate_roa_states,
            device=device,
            dtype=dtype,
        )
        
        train_utils.train_lyapunov_with_buffer(
            derivative_lyaloss=derivative_lyaloss,
            positivity_lyaloss=positivity_lyaloss,
            observer_loss=None,
            lower_limit=scaled_lower,
            upper_limit=scaled_upper,
            grid_size=grid_size,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            max_iter=cfg.train.max_iter,
            enable_wandb=cfg.train.wandb.enabled,
            derivative_ibp_ratio=cfg.loss.ibp_ratio_derivative,
            derivative_sample_ratio=cfg.loss.sample_ratio_derivative,
            positivity_ibp_ratio=0.0,
            positivity_sample_ratio=0.0,
            save_best_model=save_path,
            pgd_steps=cfg.train.pgd_steps,
            buffer_size=cfg.train.buffer_size,
            batch_size=cfg.train.batch_size,
            epochs=cfg.train.epochs,
            samples_per_iter=cfg.train.samples_per_iter,
            l1_reg=cfg.loss.l1_reg,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            V_decrease_within_roa=cfg.model.V_decrease_within_roa,
            Vmin_x_boundary_weight=cfg.loss.Vmin_x_boundary_weight,
            Vmax_x_boundary_weight=cfg.loss.Vmax_x_boundary_weight,
            Vmin_x_pgd_buffer_size=cfg.train.Vmin_x_pgd_buffer_size,
            candidate_roa_states=candidate_roa_states,
            candidate_roa_states_weight=cfg.loss.candidate_roa_states_weight,
            always_candidate_roa_regulizer=cfg.loss.always_candidate_roa_regulizer,
            logger=logger,
        )

    # ========== ROA 可视化（x 与 x_dot）==========
    if cfg.model.V_decrease_within_roa:
        # 确保 x_boundary 已经设置，用于计算 rho
        if derivative_lyaloss.x_boundary is None:
            derivative_lyaloss.x_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                lyapunov_nn,
                scaled_lower,
                scaled_upper,
                num_samples_per_boundary=cfg.train.num_samples_per_boundary,
                eps=scaled_limit,
                steps=100,
                direction="minimize",
            )
        rho = derivative_lyaloss.get_rho().item()
        fig = plt.figure()
        train_utils.plot_V_heatmap(
            fig,
            lyapunov_nn,
            rho,
            scaled_lower,
            scaled_upper,
            position_continuous.nx,
            x_boundary=derivative_lyaloss.x_boundary,
            plot_idx=[0, 1],
            mode=0.0,
        )
        plt.savefig(os.path.join(os.getcwd(), "V_roa_x.png"))
        plt.close(fig)
        logger.info("ROA 可视化已保存到: V_roa_x.png")

    # ========== 最终评估 ==========
    logger.info("\n" + "=" * 60)
    logger.info("最终评估")
    logger.info("=" * 60)

    x_equilibrium = position_continuous.x_equilibrium.unsqueeze(0).to(device)

    controller.eval()
    lyapunov_nn.eval()

    with torch.no_grad():
        u_best = controller.forward(x_equilibrium)
        V_min = lyapunov_nn.forward(x_equilibrium)

    logger.info(f"平衡点状态 x*: {x_equilibrium.squeeze().cpu().numpy()}")
    logger.info(f"最小 Lyapunov 值 V(x*): {V_min.item():.6f}")
    logger.info(f"最优控制 U_x: {u_best.item():.6f}")
    
    # 保存最终模型
    final_dir = os.path.join(os.getcwd(), "final")
    os.makedirs(final_dir, exist_ok=True)

    # 保存最优控制
    torch.save({
        'x_equilibrium': x_equilibrium.cpu(),
        'u_best': u_best.cpu(),
        'V_min': V_min.cpu(),
        'axis': 'x',
    }, os.path.join(final_dir, 'u_best_x.pt'))

    torch.save(controller.state_dict(), os.path.join(final_dir, "controller_final.pth"))
    torch.save(lyapunov_nn.state_dict(), os.path.join(final_dir, "lyapunov_final.pth"))
    torch.save(derivative_lyaloss.state_dict(), os.path.join(final_dir, "lyaloss_final.pth"))
    logger.info(f"最终模型已保存到: {final_dir}")


if __name__ == "__main__":
    main()
