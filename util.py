import numpy as np
import cv2


class KittiUtil:

    def __init__(self, filepath) -> None:
        data = self.read_calib_file(filepath)
        self.P = []
        self.P.append(data['P0'].reshape((3,4)))
        self.P.append(data['P1'].reshape((3,4)))
        self.P.append(data['P2'].reshape((3,4)))
        self.P.append(data['P3'].reshape((3,4)))
        
        Tr = data['Tr'].reshape(3,4)
        Tr = np.vstack([Tr, [0,0,0,1]])
        self.T_v2c = Tr
        

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def velo_to_cam(self, pcd, cam_id):
        out =  (self.P[cam_id] @ self.T_v2c @ pcd.T).T

        out[:, 0] = out[:, 0] / out[:, 2]
        out[:, 1] = out[:, 1] / out[:, 2]
        out[:, 2] = out[:, 2] / out[:, 2]

        return out

    def load_pcd(self, path):
        pcd = np.fromfile(path, dtype=np.float32)
        pcd = pcd.reshape(-1, 4)
        pcd[:, 3] = 1
        return pcd

    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


