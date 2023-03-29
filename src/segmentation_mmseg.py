# Remove warnings
import warnings
warnings.filterwarnings("ignore")

# Import libraries
import time, os
import rospy, rospkg
import numpy as np
import cv2
import torch

from mmseg.core.evaluation import get_palette
from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import (get_backend, 
                                         load_config,
                                         get_input_shape)

rospkg = rospkg.RosPack()
pkg_path = rospkg.get_path('semantic_segmentation_ros')
models_path = pkg_path + '/models/'
media_path = pkg_path + '/media/'

# Access the rosparams
param_model_name = rospy.get_param('model_name')
param_deploy_cfg = rospy.get_param('deploy_cfg')
param_model_cfg = rospy.get_param('model_cfg')
param_trt_model = rospy.get_param('trt_model')
param_backend = rospy.get_param('backend')

param_mmseg_path = rospy.get_param('mmseg_path')
param_mmdeploy_path = rospy.get_param('mmdeploy_path')

mmseg_config_path = param_mmseg_path + 'configs/' + param_model_name
mmdeploy_config_path = param_mmdeploy_path + 'configs/mmseg/' 


deploy_cfg_path = os.path.join(mmdeploy_config_path, param_deploy_cfg)
model_cfg_path = os.path.join(mmseg_config_path, param_model_cfg)
models_path = os.path.join(models_path, param_model_name)
trt_model_path = os.path.join(models_path, param_trt_model)
model_path = [trt_model_path]

output_path = os.path.join(models_path, 'output')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

total_time, image_no = 0, 0
test = False

print("Using device    : ", device)
print("Model name      : ", param_model_name)
print("Deploy cfg      : ", param_deploy_cfg)
print("Model cfg       : ", param_model_cfg)
print("Backend         : ", param_backend)
print("TRT Model       : ", param_trt_model)
print("Deploy cfg path : ", deploy_cfg_path)
print("Model cfg path  : ", model_cfg_path)
print("TRT Model path  : ", trt_model_path)

deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
input_shape = get_input_shape(deploy_cfg)
task_processor = build_task_processor(model_cfg, deploy_cfg, device)

# create model an inputs
model = task_processor.init_backend_model(model_path)
backend = get_backend(deploy_cfg).value

model = model.eval().to(device)
torch.backends.cudnn.benchmark = True

palette = get_palette('cityscapes')

if test:
    img = cv2.imread(media_path + '/test.jpg')
    data, _ = task_processor.create_input(img, input_shape)
    result = task_processor.run_inference(model, data)
    print(result[0].shape)
    seg = result[0]
    if seg.dtype == np.float32:
        seg = np.argmax(seg, axis=0)

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    cv2.imwrite(output_path + '/output_segmentation.png', img)
    cv2.imwrite(output_path + '/output_segmentation_seg.png', color_seg)


def segmentation_mmseg(image):
    global image_no
    global total_time
    
    start_time = time.time()
    data, _ = task_processor.create_input(image, input_shape)
    result = task_processor.run_inference(model, data)
    end_time = time.time()
    
    frame_time = end_time - start_time
    total_time += frame_time
    
    start_time = time.time()
    seg = result[0]
    
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
        
    img = color_seg.astype(np.uint8)
    end_time = time.time()
    frame_time_2 = end_time - start_time
    print("Frame no {}. Time taken for model: {:.2f}s, Time take for post-processing: {:.2f}s. ".format(image_no, frame_time, frame_time_2))
    image_no += 1

    return img
    