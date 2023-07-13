import argparse
import os, sys
from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
# parser.add_argument('--filename', required=True, help='path to sens file to read')
parser.add_argument('--filedir', required=True, help='path to sens file to read')
parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

opt = parser.parse_args()
print(opt)


def main():

  for folder_name in os.listdir(opt.filedir):
    file_name = os.path.join(opt.filedir, folder_name)
    output_name = os.path.join(opt.output_path, folder_name)
    sens_name = os.path.join(file_name,"{}.sens".format(folder_name))

    if not os.path.exists(output_name):
      os.makedirs(output_name)
    # load the data
    sys.stdout.write('loading %s...' % sens_name)
    sd = SensorData(sens_name)
    sys.stdout.write('loaded!\n')
    if opt.export_depth_images:
      sd.export_depth_images(os.path.join(output_name, 'depth'))
    if opt.export_color_images:
      sd.export_color_images(os.path.join(output_name, 'color'))
    if opt.export_poses:
      sd.export_poses(os.path.join(output_name, 'pose'))
    if opt.export_intrinsics:
      sd.export_intrinsics(os.path.join(output_name, 'intrinsic'))


if __name__ == '__main__':
    main()
