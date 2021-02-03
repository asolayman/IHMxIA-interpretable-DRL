import numpy as np
import skimage.transform
from skimage.color import rgb2gray
from vizdoom import *

base = 'scenarios/'


class DoomEnv:
    def __init__(self, args, windowDisp):
        self.channels = args.channels
        self.game = self.initialize_vizdoom(base + args.scenario + ".cfg", windowDisp)
        self.resolution = [args.img_width, args.img_height]
        self.n_actions = self.game.get_available_buttons_size()

    def preprocess(self, img):
        img = skimage.transform.resize(img, [self.resolution[0], self.resolution[1]])
        img = img.astype(np.float32)
        img *= (1.0 / 255.0)
        img = img.reshape([1, self.channels, self.resolution[0], self.resolution[1]])
        return img

    def initialize_vizdoom(self, config_file_path, test_mod=False):
        print("Initializing doom...")
        game = DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(test_mod)
        game.set_mode(Mode.ASYNC_PLAYER)
        # game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.init()
        print("Doom initialized.")

        return game

    def get_state(self):
        return self.preprocess(self.game.get_state().screen_buffer.transpose((1, 0)))

    def get_raw_state(self):
        return self.game.get_state().screen_buffer
