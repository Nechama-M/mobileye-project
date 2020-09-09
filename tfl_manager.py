import os
import numpy as np
from PIL import Image
from part1 import run_attention
from part2 import init_data_set
from tensorflow.keras.models import load_model
from part3 import SFM
from part3.SFM_standAlone import FrameContainer, visualize
from visual import visualize


class TFL_Manager():
    def __init__(self, focal, pp):
        self.focal = focal
        self.pp = pp
        self.frame = []
        self.prev_container = None
        self.current_container = None
        self.model = load_model(os.path.join("part2", "model.h5"))

    def run_part1(self):
        image = np.array(self.current_container.img)
        x_red, y_red, x_green, y_green = run_attention.find_tfl_lights(image)
        candidates = []
        auxiliary = []

        for i in range(len(x_red)):
            candidates.append([x_red[i], y_red[i]])
            auxiliary.append("red")

        for i in range(len(x_green)):
            flag = True
            for j in candidates:
                if (abs(y_green[i] - j[1]) < 10) and (abs(x_green[i] - j[0]) < 10):
                    flag = False
            if flag:
                candidates.append([x_green[i], y_green[i]])
                auxiliary.append("green")

        return candidates, auxiliary

    def is_traffic_light(self, image):
        crop_shape = (81, 81)
        test_image = image.reshape([-1] + list(crop_shape) + [3])
        predictions = self.model.predict(test_image)
        predicted_label = np.argmax(predictions, axis=-1)
        if predicted_label[0] == 1:
            return True
        else:
            return False

    def run_part2(self, candidates, auxiliary):
        image = Image.open(self.current_container.img_path)
        traffic_lights = []
        tfl_auxiliary = []
        for i in range(len(candidates)):
            cropped_image = init_data_set.crop_image_around_coordinate(image, (candidates[i][1], candidates[i][0]))

            if self.is_traffic_light(np.array(cropped_image)):
                traffic_lights.append(candidates[i])
                tfl_auxiliary.append(auxiliary[i])

        return traffic_lights, tfl_auxiliary

    def run_part3(self, EM):
        self.current_container.EM = EM
        self.current_container = SFM.calc_TFL_dist(self.prev_container, self.current_container, self.focal, self.pp)

    def run(self, frame, EM):
        self.current_container = FrameContainer(frame)
        candidates, auxiliary = self.run_part1()
        self.current_container.traffic_light, self.current_container.auxiliary = self.run_part2(candidates,
                                                                                                auxiliary)
        try:
            # sanity: make sure part2 returns not more than part1 candidates
            assert len(self.current_container.traffic_light) <= len(candidates)
        except AssertionError:
            self.current_container.traffic_light, self.current_container.auxiliary = candidates, auxiliary
        if EM is not None:
            self.run_part3(EM)

        visualize(candidates, auxiliary, self.prev_container, self.current_container, self.focal, self.pp)
        self.prev_container = self.current_container
        self.current_container = None
