import os
import numpy as np
import mujoco
from mujoco import MjModel, MjData
from ur10_sim.utils.ik_arm import IKArm
from ur10_sim.utils.util import calculate_arm_Te

# 路径设置
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../envs/xmls/ur10.xml"
)

def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 初始关节角
    q0 = np.random.uniform(low=-np.pi, high=np.pi, size=model.nv)
    data.qpos[:] = q0
    mujoco.mj_fwdPosition(model, data)

    # 当前末端位姿
    Te = calculate_arm_Te(data.body("attachment").xpos, data.body("attachment").xquat)
    print("当前末端位姿 Te:\n", Te)

    # 构造目标末端位姿（例如在z轴方向移动0.1m）
    Tep = Te.copy()
    Tep[2, 3] += 0.1

    print("目标末端位姿 Tep:\n", Tep)

    # 创建IK求解器
    ik_solver = IKArm(solver_type='QP', tol=1e-4, ilimit=100)

    # 求解逆运动学
    q_sol, success, iterations, error, jl_valid, total_t = ik_solver.solve(model, data, Tep, q0)

    print("IK求解成功:", success)
    print("求解关节角:", q_sol)
    print("迭代次数:", iterations)
    print("残差误差:", error)
    print("关节限位有效:", jl_valid)
    print("总耗时(s):", total_t)

    # 检查正向运动学是否到达目标
    data.qpos[:] = q_sol
    mujoco.mj_fwdPosition(model, data)
    Te_check = calculate_arm_Te(data.body("attachment").xpos, data.body("attachment").xquat)
    print("IK解对应的末端位姿:\n", Te_check)
    print("末端位置误差:", np.linalg.norm(Te_check[:3, 3] - Tep[:3, 3]))

if __name__ == "__main__":
    main()
