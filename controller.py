import pickle
import numpy as np

from tfl_manager import TFL_Manager


class Controller:
    def __init__(self, pls):
        self.pls = pls
        self.pkl_path, self.frames = self.get_frames()
        self.focal, self.pp, self.EM_data = self.load_pkl()

    def get_frames(self):
        with open(self.pls, "r", encoding='utf8') as file:
            data = file.readlines()
            frames = []
            pkl_path = data[0][:-1]
            for i in data[1:]:
                frames.append(i[:-1] if i[-1] == '\n' else i)

            return pkl_path, frames

    def load_pkl(self):
        with open(self.pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')

        focal = data['flx']
        pp = data['principle_point']

        return focal, pp, data

    def run(self):
        START_FRAME = int(self.frames[0][-22: -16])
        tfl_manager = TFL_Manager(self.focal, self.pp)
        for i in range(len(self.frames)):
            EM = np.eye(4)
            if i > 0:
                for j in range(START_FRAME + i - 1, START_FRAME + i):
                    EM = np.dot(self.EM_data['egomotion_' + str(j) + '-' + str(j + 1)], EM)
            else:
                EM = None
            tfl_manager.run(self.frames[i], EM)


if __name__ == '__main__':
    controller = Controller("pls.pls")
    controller.run()
