import argparse
import os
import sys
import glob

import numpy as np
import pandas as pd

# to make run from console for module import
sys.path.append(os.path.abspath('..'))

from src.main.config import Config
from src.main.model import Model
from src.visualise.trimesh_renderer import TrimeshRenderer
from src.visualise.vis_util import preprocess_image, visualize


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def join_csv():
  path = 'output/csv/'                   
  all_files = glob.glob(os.path.join(path, "*.csv"))
  all_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
  df_from_each_file = (pd.read_csv(f) for f in all_files)
  concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)

  concatenated_df['frame'] = concatenated_df.index+1
  concatenated_df.to_csv("output/csv_joined/csv_joined.csv", index=False)

def main(img_path, json_path=None, input_shape=224):
    # initialize model
    model = Model()
    original_img, input_img, params = preprocess_image(img_path, input_shape, json_path)

    result = model.detect(input_img)

    cam = np.squeeze(result['cam'].numpy())[:3]
    vertices = np.squeeze(result['vertices'].numpy())
    joints = np.squeeze(result['kp2d'].numpy())
    joints = ((joints + 1) * 0.5) * params['img_size']
    joints3d = np.squeeze(result['kp3d'].numpy())

    renderer = TrimeshRenderer()
    visualize(renderer, img_path, original_img, params, vertices, cam, joints)

    # export to CSV
    joints_names = ['Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z',
                   'Knee.R_x', 'Knee.R_y', 'Knee.R_z',
                   'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                   'Hip.L_x', 'Hip.L_y', 'Hip.L_z',
                   'Knee.L_x', 'Knee.L_y', 'Knee.L_z', 
                   'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
                   'Wrist.R_x', 'Wrist.R_y', 'Wrist.R_z', 
                   'Elbow.R_x', 'Elbow.R_y', 'Elbow.R_z', 
                   'Shoulder.R_x', 'Shoulder.R_y', 'Shoulder.R_z', 
                   'Shoulder.L_x', 'Shoulder.L_y', 'Shoulder.L_z',
                   'Elbow.L_x', 'Elbow.L_y', 'Elbow.L_z',
                   'Wrist.L_x', 'Wrist.L_y', 'Wrist.L_z', 
                   'Neck_x', 'Neck_y', 'Neck_z', 
                   'Head_x', 'Head_y', 'Head_z', 
                   'Nose_x', 'Nose_y', 'Nose_z', 
                   'Eye.L_x', 'Eye.L_y', 'Eye.L_z', 
                   'Eye.R_x', 'Eye.R_y', 'Eye.R_z', 
                   'Ear.L_x', 'Ear.L_y', 'Ear.L_z', 
                   'Ear.R_x', 'Ear.R_y', 'Ear.R_z']
    
    joints_export = pd.DataFrame(joints3d.reshape(1,57), columns=joints_names)
    joints_export.index.name = 'frame'
    
    joints_export.iloc[:, 1::3] = joints_export.iloc[:, 1::3]*-1
    joints_export.iloc[:, 2::3] = joints_export.iloc[:, 2::3]*-1

    
    hipCenter = joints_export.loc[:][['Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                                      'Hip.L_x', 'Hip.L_y', 'Hip.L_z']]
                                      
    joints_export['hip.Center_x'] = hipCenter.iloc[0][::3].sum()/2
    joints_export['hip.Center_y'] = hipCenter.iloc[0][1::3].sum()/2
    joints_export['hip.Center_z'] = hipCenter.iloc[0][2::3].sum()/2
    
    joints_export.to_csv("output/csv/"+os.path.splitext(os.path.basename(img_path))[0]+".csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo HMR2.0')

    parser.add_argument('--img_path', required=False, default='images/coco1.png')
    parser.add_argument('--json_path', required=False)
    parser.add_argument('--model', required=False, default='base_model', help="model from logs folder")
    parser.add_argument('--setting', required=False, default='paired(joints)', help="setting of the model")
    parser.add_argument('--joint_type', required=False, default='cocoplus', help="<cocoplus|custom>")
    parser.add_argument('--init_toes', required=False, default=False, type=str2bool,
                        help="only set to True when joint_type=cocoplus")

    args = parser.parse_args()
    if args.init_toes:
        assert args.joint_type, "Only init toes when joint type is cocoplus!"


    class DemoConfig(Config):
        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath('logs/{}/{}'.format(args.setting, args.model))
        INITIALIZE_CUSTOM_REGRESSOR = args.init_toes
        JOINT_TYPE = args.joint_type

    config = DemoConfig()

    main(args.img_path, args.json_path, config.ENCODER_INPUT_SHAPE[0])

    join_csv()