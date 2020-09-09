from tfl_manager import TFL_Manager


class Controller:
    def __init__(self, pls):
        self.pls = pls

    def get_frames(self):
        with open(self.pls, "r", encoding='utf8') as file:
            data = file.readlines()
            frames = []
            pkl_path = data[0][:-1]
            for i in data[1:]:
                frames.append(i[:-1] if i[-1] == '\n' else i)

            return pkl_path, frames

    def init(self):
        pkl_path, frames = self.get_frames()
        tfl_manager = TFL_Manager(frames, pkl_path)
        tfl_manager.run()


if __name__ == '__main__':
    controller = Controller("pls.pls")
    controller.init()