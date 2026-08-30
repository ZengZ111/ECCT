import os
from rknn.api import RKNN

TARGET_PLATFORM = 'rk3588'
DO_QUANTIZATION = False

def convert_onnx_to_rknn(onnx_path, rknn_path, mean_values, std_values):
    rknn = RKNN(verbose=True)
    rknn.config(mean_values=mean_values, std_values=std_values, target_platform=TARGET_PLATFORM)
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"load onnx failed: {onnx_path}")
        rknn.release()
        return False
    ret = rknn.build(do_quantization=DO_QUANTIZATION)
    if ret != 0:
        print(f"build failed: {onnx_path}")
        rknn.release()
        return False
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"export failed: {rknn_path}")
        rknn.release()
        return False
    print(f"✅ {rknn_path} 转换成功")
    rknn.release()
    return True

def main():
    os.makedirs('./rknn_models', exist_ok=True)
    # Backbone 127
    convert_onnx_to_rknn('./models/onnx/nanotrack_backbone_127.onnx',
                         './rknn_models/backbone_127.rknn',
                         [[0,0,0]], [[1,1,1]])
    # Backbone 255
    convert_onnx_to_rknn('./models/onnx/nanotrack_backbone_255.onnx',
                         './rknn_models/backbone_255.rknn',
                         [[0,0,0]], [[1,1,1]])
    # Head
    convert_onnx_to_rknn('./models/onnx/nanotrack_head.onnx',
                         './rknn_models/head.rknn',
                         [[0]*96, [0]*96], [[1]*96, [1]*96])

if __name__ == '__main__':
    main()