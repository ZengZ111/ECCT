# rknn_client.py
# 纯 TCP 客户端，无任何应用层网络模拟

import socket
import json
import cv2

class VLMClient:
    def __init__(self, pc_ip, pc_port=9999):
        """
        初始化 VLM 客户端
        :param pc_ip: 网关 IP（WSL2 的 IP，如 192.168.0.24）
        :param pc_port: 网关端口（默认 9999）
        """
        self.pc_ip = pc_ip
        self.pc_port = pc_port
        self.sock = None

    def connect(self):
        """建立 TCP 连接到网关"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.pc_ip, self.pc_port))

    def filter_boxes(self, image, boxes, description, overscan=50):
        """
        发送裁剪图批次到网关进行语义验证
        :param image: 原始图像 (numpy array)
        :param boxes: 边界框列表，每个框为 [x1, y1, x2, y2, conf]
        :param description: 文本描述
        :param overscan: 裁剪外扩像素
        :return: (matches, justifications, sent_bytes)
        """
        crops_jpegs = []
        h, w = image.shape[:2]
        for (x1, y1, x2, y2, conf) in boxes:
            x1_e = int(max(0, x1 - overscan))
            y1_e = int(max(0, y1 - overscan))
            x2_e = int(min(w, x2 + overscan))
            y2_e = int(min(h, y2 + overscan))
            crop = image[y1_e:y2_e, x1_e:x2_e]
            if crop.size == 0:
                continue
            _, jpeg_buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            crops_jpegs.append(jpeg_buf.tobytes())

        if not crops_jpegs:
            return [], [], 0

        # 构建二进制协议包
        magic = 0xCAFEBABE
        req_type = 0x01  # 裁剪图模式
        desc_bytes = description.encode('utf-8')
        desc_len = len(desc_bytes)
        num_crops = len(crops_jpegs)

        payload = bytearray()
        payload += magic.to_bytes(4, 'big')
        payload += req_type.to_bytes(1, 'big')
        payload += desc_len.to_bytes(4, 'big')
        payload += desc_bytes
        payload += num_crops.to_bytes(4, 'big')
        for jpg_data in crops_jpegs:
            payload += len(jpg_data).to_bytes(4, 'big')
            payload += jpg_data

        # 发送数据（无模拟）
        self.sock.sendall(payload)
        sent_bytes = len(payload)

        # 接收响应
        resp = self._recv_json()
        if resp is None:
            return [], [], sent_bytes
        return resp.get("matches", []), resp.get("justifications", []), sent_bytes

    def detect_and_filter(self, image, category, description, overscan=50):
        """
        发送整帧图像到网关进行目标检测与语义验证
        :param image: 原始图像 (numpy array)
        :param category: 短类别（如 'person'）
        :param description: 文本描述
        :param overscan: 暂未使用（保留接口一致性）
        :return: (boxes_with_conf, justifications, sent_bytes)
        """
        _, jpeg_buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg_data = jpeg_buf.tobytes()

        magic = 0xCAFEBABE
        req_type = 0x00  # 整帧模式
        cat_bytes = category.encode('utf-8')
        cat_len = len(cat_bytes)
        desc_bytes = description.encode('utf-8')
        desc_len = len(desc_bytes)

        payload = bytearray()
        payload += magic.to_bytes(4, 'big')
        payload += req_type.to_bytes(1, 'big')
        payload += cat_len.to_bytes(4, 'big')
        payload += cat_bytes
        payload += desc_len.to_bytes(4, 'big')
        payload += desc_bytes
        payload += len(jpg_data).to_bytes(4, 'big')
        payload += jpg_data

        # 发送数据（无模拟）
        self.sock.sendall(payload)
        sent_bytes = len(payload)

        resp = self._recv_json()
        if resp is None:
            return [], [], sent_bytes
        boxes = resp.get("boxes", [])
        justifications = resp.get("justifications", [])
        return boxes, justifications, sent_bytes

    def _recv_json(self):
        """
        接收网关返回的 JSON 响应（4字节长度 + JSON数据）
        """
        raw_len = self.sock.recv(4)
        if not raw_len:
            raise socket.timeout("Connection closed")

        msg_len = int.from_bytes(raw_len, 'big')
        chunks = []
        bytes_recd = 0
        while bytes_recd < msg_len:
            chunk = self.sock.recv(min(msg_len - bytes_recd, 4096))
            if not chunk:
                raise RuntimeError("Connection broken")
            chunks.append(chunk)
            bytes_recd += len(chunk)
        return json.loads(b''.join(chunks).decode('utf-8'))

    def close(self):
        if self.sock:
            self.sock.close()