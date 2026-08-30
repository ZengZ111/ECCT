import socket
import json
import io
import sys
from PIL import Image
from loguru import logger
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cloud_track.rpc_communication.rpc_wrapper import RpcWrapper

class VLMGateway:
    def __init__(self, backend_ip, backend_port, listen_port=9999):
        self.backend = RpcWrapper(backend_ip, backend_port)
        self.listen_port = listen_port
        self.server_socket = None

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.listen_port))
        self.server_socket.listen(5)
        logger.info(f"VLM Gateway (Binary Protocol) listening on port {self.listen_port}")

        while True:
            conn, addr = self.server_socket.accept()
            logger.info(f"Connection from {addr}")
            conn.settimeout(5)
            with conn:
                while True:
                    try:
                        header = self._recv_exact(conn, 5)
                        if not header:
                            break
                        magic = int.from_bytes(header[:4], 'big')
                        req_type = header[4]

                        if magic != 0xCAFEBABE:
                            logger.error(f"Invalid magic: {hex(magic)}")
                            break

                        if req_type == 0x00:  # 整帧
                            # 读取 category (短词)
                            cat_len = int.from_bytes(self._recv_exact(conn, 4), 'big')
                            cat_bytes = self._recv_exact(conn, cat_len)
                            category = cat_bytes.decode('utf-8')

                            # 读取 description (长句)
                            desc_len = int.from_bytes(self._recv_exact(conn, 4), 'big')
                            desc_bytes = self._recv_exact(conn, desc_len)
                            description = desc_bytes.decode('utf-8')

                            # 读取图像
                            img_len = int.from_bytes(self._recv_exact(conn, 4), 'big')
                            img_bytes = self._recv_exact(conn, img_len)
                            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')

                            # 调用后端，category 给 GroundingDINO，description 给 VLM
                            result = self.backend.run_inference(
                                img_pil, category, description, False, True
                            )
                            boxes_filt = result[2]
                            justifications = result[4]
                            scores = result[3]

                            boxes_with_conf = []
                            if boxes_filt is not None:
                                if hasattr(boxes_filt, 'tolist'):
                                    boxes_filt = boxes_filt.tolist()
                                for box, score in zip(boxes_filt, scores):
                                    if len(box) == 4:
                                        boxes_with_conf.append(box + [float(score)])
                                    else:
                                        boxes_with_conf.append(box)
                            response = {"boxes": boxes_with_conf, "justifications": justifications}
                            self._send_json(conn, response)

                        elif req_type == 0x01:  # 裁剪图
                            desc_len = int.from_bytes(self._recv_exact(conn, 4), 'big')
                            desc_bytes = self._recv_exact(conn, desc_len)
                            description = desc_bytes.decode('utf-8')

                            num_crops = int.from_bytes(self._recv_exact(conn, 4), 'big')
                            images = []
                            for _ in range(num_crops):
                                crop_len = int.from_bytes(self._recv_exact(conn, 4), 'big')
                                crop_bytes = self._recv_exact(conn, crop_len)
                                img_pil = Image.open(io.BytesIO(crop_bytes)).convert('RGB')
                                images.append(img_pil)

                            if num_crops == 0:
                                matches = []
                                justifications = []
                            else:
                                results = self.backend.run_vlm_only_batch(images, description)
                                matches = [r[0] for r in results]
                                justifications = [r[1] for r in results]

                            response = {"matches": matches, "justifications": justifications}
                            self._send_json(conn, response)

                        else:
                            logger.error(f"Unknown req_type: {req_type}")
                            break

                    except socket.timeout:
                        logger.warning("Timeout")
                        break
                    except ConnectionResetError:
                        logger.warning("Connection reset")
                        break
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        break

    def _recv_exact(self, conn, n):
        data = bytearray()
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def _send_json(self, conn, data):
        json_str = json.dumps(data)
        msg = len(json_str).to_bytes(4, 'big') + json_str.encode('utf-8')
        conn.sendall(msg)


if __name__ == "__main__":
    gateway = VLMGateway(backend_ip="http://aaa.bbb.ccc.ddd", backend_port=3000, listen_port=9999)
    gateway.start()