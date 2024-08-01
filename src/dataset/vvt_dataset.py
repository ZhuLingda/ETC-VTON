import os.path as osp
from glob import glob

from .vvton_dataset import VVTONDataset


# Testing only
class VVTListDataset(VVTONDataset):
    def __init__(self, dataroot, image_size=256, mode="test",frames_num=3, is_pair=False):
        super().__init__(dataroot, image_size=256, mode="test",frames_num=frames_num, is_pair=is_pair)
        self.data_list = f"{self.root}/test_person_clothes_pose_tuple.txt"
        self.image_paths = []
        self.cloth_paths = []
        self.is_pair = is_pair
        self.load_file_paths()
        
    def load_file_paths(self):

        # make list of
        # cloth <---> image
        with open(self.data_list, "r") as f:
            for line in f:
                # image dir should be our GFLA result
                # we need to Dress cloth_id to image_dir
                image_dir, cloth_id, pose_dir = line.strip().split()
                if self.is_pair:
                    cloth_id = image_dir
                image_paths = sorted(
                    glob(f"{self.root}/test_frames/{image_dir}/*.png")
                )
                remain_image_path_list = []
                for img_path in image_paths:
                    pose_path = img_path.replace('frames', 'frames_keypoint').replace('.png', '_keypoints.json')
                    if osp.exists(pose_path):
                        remain_image_path_list.append(img_path)

                # copies the same source cloth_file for the number of test frames
                image_paths = remain_image_path_list
                cloth_file = glob(
                    f"{self.root}/clothes_person/{cloth_id}/*cloth*"
                )[0]
                cloth_paths = [cloth_file] * len(image_paths)


                assert len(image_paths) == len(
                    cloth_paths
                ), f"lens don't match on {image_dir}"
                self.image_paths.extend(image_paths)
                self.cloth_paths.extend(cloth_paths)

    def __len__(self):
        return len(self.image_paths)

    def get_person_image_path(self, index):
        return self.image_paths[index]

    def get_input_cloth_path(self, index):
        return self.cloth_paths[index]

    def get_input_cloth_name(self, index):
        # in test stage, use the folder id of the person. because the clothes will match the person
        image_path = self.get_person_image_path(index)
        folder_id = VVTONDataset.extract_video_id(image_path)
        cloth_path = self.get_input_cloth_path(index)
        base_cloth_name = osp.basename(cloth_path)
        frame_name = osp.basename(self.get_person_image_name(index))
        # e.g. 4he21d00f-g11/4he21d00f-g11@10=cloth_front.jpg
        name = osp.join(folder_id, f"{base_cloth_name}.FOR.{frame_name}")
        return name
