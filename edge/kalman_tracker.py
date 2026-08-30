# kalman_tracker.py
import numpy as np

class KalmanBoxTracker:
    """
    基于恒速模型（Constant Velocity）的卡尔曼滤波器
    状态向量: [cx, cy, w, h, vx, vy, vw, vh]^T
    测量向量: [cx, cy, w, h]^T
    """
    def __init__(self, bbox, dt=1.0):
        """
        bbox: [x1, y1, x2, y2]  (归一化后的框)
        """
        # 初始化状态 (8维)
        self.x = np.zeros((8, 1), dtype=np.float32)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.x[0:4, 0] = [cx, cy, w, h]
        self.x[4:8, 0] = [0, 0, 0, 0]

        # 状态转移矩阵 F
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i+4] = dt

        # 测量矩阵 H
        self.H = np.eye(4, 8, dtype=np.float32)

        # 过程噪声协方差 Q
        self.Q = np.eye(8, dtype=np.float32) * 0.01
        self.Q[4:8, 4:8] *= 0.1

        # 测量噪声协方差 R
        self.R = np.eye(4, dtype=np.float32) * 5.0
        self.R[2, 2] = 10.0
        self.R[3, 3] = 10.0

        # 协方差矩阵 P
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:8, 4:8] *= 100.0

        # 历史残差（用于动态阈值）
        self.residual_history = []
        self.max_history = 30

    def predict(self):
        """预测下一帧状态"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_state()

    def update(self, bbox):
        """更新观测值"""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        residual_norm = np.linalg.norm(y[0:2])
        self.residual_history.append(residual_norm)
        if len(self.residual_history) > self.max_history:
            self.residual_history.pop(0)

        return self.get_state(), residual_norm

    def get_state(self):
        return self.x[0:4].flatten()

    def get_predicted_bbox(self):
        cx, cy, w, h = self.x[0:4].flatten()
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    def is_anomaly(self, bbox, threshold_factor=2.5, min_threshold=30.0):
        """
        判断当前测量框是否为异常
        threshold_factor: 动态阈值系数（标准差倍数）
        min_threshold: 最小阈值（像素），防止初始阶段误判
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        pred_cx, pred_cy = self.x[0, 0], self.x[1, 0]
        dist = np.hypot(cx - pred_cx, cy - pred_cy)

        if len(self.residual_history) > 10:
            mean_res = np.mean(self.residual_history)
            std_res = np.std(self.residual_history)
            threshold = max(mean_res + threshold_factor * std_res, min_threshold)
        else:
            threshold = min_threshold

        return dist > threshold