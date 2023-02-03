import os
import sys
import pickle
import yaml
import numpy as np
from pathlib import Path
from easydict import EasyDict
from skimage import io

import pcdet.datasets.kitti.kitti_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.datasets.dataset import DatasetTemplate

class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        #[N, 1]
        return pts_valid_flag

    def create_velodynes(self, num_workers=1, has_label=True, count_inside_pts=True, sample_id_list=None, new_path=None):
        import concurrent.futures as futures
        
        def process_single_scene(sample_idx):
            print('---------process_single_scene -->{} set, idx: {}---------'.format(self.split, sample_idx))
            calib = self.get_calib(sample_idx)
            # P2->(4, 4)
            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            filename = os.path.join(new_path, str(sample_idx) + ".txt")
            # print("current file name is {}\n".format(filename))
            
            with open(filename, mode="w", encoding='utf-8') as f:
                if has_label:
                    obj_list = self.get_label(sample_idx)
                    annotations = {}
                    annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                    # annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                    # annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                    # annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                    # annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                    annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                    annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                    annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                    # annotations['score'] = np.array([obj.score for obj in obj_list])
                    annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                    num_gt = len(annotations['name'])
                    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                    annotations['index'] = np.array(index, dtype=np.int32)

                    loc = annotations['location'][:num_objects]
                    dims = annotations['dimensions'][:num_objects]
                    rots = annotations['rotation_y'][:num_objects]
                    # (N, 3): (N, 3)->(N, 3)
                    loc_lidar = calib.rect_to_lidar(loc)
                    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                    # update h
                    loc_lidar[:, 2] += h[:, 0] / 2
                    # (N, 3+1+1+1+ 1). rots dimension = x, y, z, dx, dy, dz, heading
                    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar
                    
                    '''
                    debug
                    ## print(gt_boxes_lidar)
                    points = self.get_lidar(sample_idx)
                    # draw_points_with_box(points, gt_boxes_lidar)
                    ## print("gt_boxes_lidar is {}".format(gt_boxes_lidar))
                    ## for i in range(num_objects):
                    ##    print("curr {} name is {}".format(i, annotations['name'][i]))
                    '''
                    anno = annotations
                    for i in range(num_objects):
                        '''
                            str(anno['name'][i]) + ' '\
                            + str(anno['truncated'][i]) + ' '\
                            + str(anno['occluded'][i]) + ' '\
                            + str(anno['alpha'][i]) + ' '\
                            + str(anno['bbox'][i][0]) + ' '\
                            + str(anno['bbox'][i][1]) + ' '\
                            + str(anno['bbox'][i][2]) + ' '\
                            + str(anno['bbox'][i][3]) + ' '\
                        '''
                        w_s = str(anno['gt_boxes_lidar'][i][0]) + ' '\
                            + str(anno['gt_boxes_lidar'][i][1]) + ' '\
                            + str(anno['gt_boxes_lidar'][i][2]) + ' '\
                            + str(anno['gt_boxes_lidar'][i][3]) + ' '\
                            + str(anno['gt_boxes_lidar'][i][4]) + ' '\
                            + str(anno['gt_boxes_lidar'][i][5]) + ' '\
                            + str(anno['gt_boxes_lidar'][i][6]) + ' '\
                            + str(anno['name'][i]) + '\n'
                        ## print("current object info is {}\n".format(w_s))
                        f.write(w_s)
        
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        # print("sample id list is: {}".format(sample_id_list))
        num_workers = 1
        # with futures.ThreadPoolExecutor(num_workers) as executor:
        #     infos = executor.map(process_single_scene, sample_id_list)

        for idx in sample_id_list:
            process_single_scene(idx)

def check_folder(file_path):
    if(os.path.exists(file_path) == False):
        os.mkdir(file_path)
        print("mkdir {}".format(file_path))

def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=1):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'
    new_velodyne_path = save_path / 'lidar_label_in_9'
    check_folder(new_velodyne_path)
    
    print('---------------Start to convert labels---------------')
    # save train files
    train_path = os.path.join(new_velodyne_path, "train")
    check_folder(train_path)
    dataset.set_split(train_split)
    dataset.create_velodynes(num_workers=workers, has_label=True, count_inside_pts=True, new_path=train_path)
    
    # save val files
    val_path = os.path.join(new_velodyne_path, "val")
    check_folder(val_path)
    dataset.set_split(val_split)
    dataset.create_velodynes(num_workers=workers, has_label=True, count_inside_pts=True, new_path=val_path)
    return 

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.exit("Too few arguments!\ncmd: python cvt_label_15to9.py ../../../tools/cfgs/dataset_configs/kitti_dataset.yaml")

    cfg_file_path = sys.argv[1]
    dataset_cfg = EasyDict(yaml.safe_load(open(cfg_file_path)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

    print('config file is: {}'.format(cfg_file_path))
    if len(sys.argv) > 1:

        print("ROOT_DIR is: {}".format(ROOT_DIR))
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti', #src dir
            save_path=ROOT_DIR / 'data' / 'custom' #dst dir
        )
    print('-----------Convert label from kitti to custom format Done------------')
