import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from part1 import run_attention
from part2 import init_data_set
from tensorflow.keras.models import load_model

from part3 import SFM
from part3.SFM_standAlone import FrameContainer, visualize


class TFL_Manager():
    def __init__(self, frames, pkl_path):
        self.frames = frames
        self.pkl_path = pkl_path
        self.prev_container = None
        self.current_container = FrameContainer(frames[0])
        self.model = load_model(os.path.join("part2", "model.h5"))

    def run_part1(self):
        image = np.array(self.current_container.img)
        x_red, y_red, x_green, y_green = run_attention.find_tfl_lights(image)
        candidates = []
        auxiliary = []

        for i in range(len(x_red)):
            candidates.append([y_red[i], x_red[i]])
            auxiliary.append("red")

        for i in range(len(x_green)):
            flag = True
            for j in candidates:
                if (abs(y_green[i] - j[0]) < 81) and (abs(x_green[i] - j[1]) < 81):
                    flag = False
            if flag:
                candidates.append([y_green[i], x_green[i]])
                auxiliary.append("green")

        return candidates, auxiliary

    def is_traffic_light(self, image):
        crop_shape = (81, 81)
        test_image = image.reshape([-1] + list(crop_shape) + [3])
        predictions = self.model.predict(test_image)
        predicted_label = np.argmax(predictions, axis=-1)
        if predicted_label[0] == 1:
            return True

        return False

    def run_part2(self, candidates, auxiliary):
        print(candidates)
        image = Image.open(self.current_container.img_path)
        traffic_lights = []
        tfl_auxiliary = []
        for i in range(len(candidates)):
            cropped_image = init_data_set.crop_image_around_coordinate(image, candidates[i])

            if self.is_traffic_light(np.array(cropped_image)):
                traffic_lights.append(candidates[i])
                tfl_auxiliary.append(auxiliary[i])

        return traffic_lights, tfl_auxiliary

    def run_part3(self):
        with open(self.pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')

        focal = data['flx']
        pp = data['principle_point']
        EM = np.eye(4)
        for i in range(self.prev_container.img_id, self.current_container.img_id):
            EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
        self.current_container.EM = EM
        self.current_container = SFM.calc_TFL_dist(self.prev_container, self.current_container, focal, pp)
        self.visualize(focal, pp)

    def run(self):
        candidates, auxiliary = self.run_part1()
        self.current_container.traffic_light, self.current_container.auxiliary = self.run_part2(candidates, auxiliary)
        self.prev_container = FrameContainer(self.frames[0])
        self.prev_container = self.current_container
        self.current_container = None

        for frame in self.frames[1:]:
            self.current_container = FrameContainer(frame)
            candidates, auxiliary = self.run_part1()
            self.current_container.traffic_light, self.current_container.auxiliary = self.run_part2(candidates[:],
                                                                                                    auxiliary[:])
            self.run_part3()

            self.prev_container = self.current_container
            self.current_container = None

    def visualize(self, focal, pp):
        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(self.prev_container, self.current_container,
                                                                            focal, pp)
        norm_rot_pts = SFM.rotate(norm_prev_pts, R)
        rot_pts = SFM.unnormalize(norm_rot_pts, focal, pp)
        foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), focal, pp))

        fig, (curr_sec, prev_sec) = plt.subplots(1, 2, figsize=(12, 6))
        prev_sec.set_title('prev(' + str(self.prev_container.img_id) + ')')
        prev_sec.imshow(self.prev_container.img)
        prev_p = self.prev_container.traffic_light[:]
        print("prev_p {}  type {}".format(prev_p,type(prev_p)))
        prev_sec.plot(prev_p[:][0], prev_p[:][1], 'b+')

        curr_sec.set_title('curr(' + str(self.current_container.img_id) + ')')
        curr_sec.imshow(self.current_container.img)
        curr_p = self.current_container.traffic_light
        curr_sec.plot(curr_p[:][0], curr_p[:][1], 'b+')

        for i in range(len(curr_p)):
            curr_sec.plot([foe[0], curr_p[i][1]], [foe[1], curr_p[i][0]], 'b')
            if self.current_container.valid[i]:
                curr_sec.text(curr_p[i][1], curr_p[i][0],
                              r'{0:.1f}'.format(self.current_container.traffic_lights_3d_location[i, 2]), color='r')
        curr_sec.plot(foe[1], foe[0], 'r+')
        curr_sec.plot(rot_pts[:, 1], rot_pts[:, 0], 'g+')
        plt.show()
