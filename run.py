import argparse
import torch
from PIL import Image, ImageFilter, ImageEnhance

from utils import embedding2csv
from agent import DQNAgent
from env import DoomEnv
import numpy as np

from moviepy.editor import ImageSequenceClip

import umap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


def parsearg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scenario', type=str, default="basic", help='vizdoom scenario')
    parser.add_argument('--test_mod', type=bool, default=False, help=' test or train model')

    parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
    parser.add_argument('--learning_steps_per_epoch', type=int, default=2000, help='number of steps per epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.00025, help='adam: learning rate')
    parser.add_argument('--discount_factor', type=float, default=0.9, help='adam: learning rate')
    parser.add_argument('--epsilon', type=float, default=0.8, help='exploration rate')
    parser.add_argument('--decay_start', type=int, default=1, help='exploration rate decay start')
    parser.add_argument('--memory_size', type=int, default=10000, help='size of the memory replay')

    parser.add_argument('--channels', type=int, default=1, help='channels of image')
    parser.add_argument('--img_height', type=int, default=64, help='size of image height')
    parser.add_argument('--img_width', type=int, default=112, help='size of image width')
    parser.add_argument('--frame_repeat', type=int, default=6, help='frame skip')

    parser.add_argument('--model_savefile', type=str, default="model.pth", help='where the saved model will be')

    return parser.parse_args()


def run():
    print("Making videos")
    imgs = []
    action_array = []
    latent_array = []
    for epoch in range(args.n_epochs):
        env.game.new_episode()
        print("\nEpisode %d\n-------" % (epoch + 1))

        while not env.game.is_episode_finished():
            state = env.get_state()

            # TODO: Change this function to return a correct grad
            best_action_index, grads, latent = agent.get_best_action_wGrad(state)
            action_array.append(best_action_index)
            latent_array.append(latent.squeeze(0).detach().numpy())
            
            env.game.make_action(agent.actions[best_action_index], args.frame_repeat)
            state = np.squeeze(state)

            # Cancel normalization
            state *= 254 * 254
            state = state.astype(np.int8)

            nb = len(imgs)

            saliency = Image.fromarray(grads, 'L')
            saliency.save("images/saliency_" + str(nb) + ".jpg")

            # grads = format_saliency(saliency)

            # If You want to enhance this display you can do
            grads = format_saliency_bprop(saliency)

            imgs.append(merge_img(state, grads, nb))

    make_movie(imgs, "video/video_" + str(epoch + 1) + ".mp4")
    
    embedding = umap.UMAP(n_neighbors=7, min_dist=0.4, metric='correlation').fit_transform(latent_array)
    embedding2csv(embedding, action_array)


def make_movie(imgs, filename):
    clip = ImageSequenceClip(imgs, fps=int(30 / args.frame_repeat))
    clip.write_videofile(filename)
    clip.write_gif(filename.split('.')[-2]+'.gif')


"""
Here we optimize the display of saliency maps to make them more readable 
(we add some color)
"""


def format_saliency_bprop(img):
    img = img.convert("RGBA")
    datas = img.getdata()

    base = (80, 80, 80, 255)
    core = (120, 120, 120, 255)
    hidden = (0, 0, 0, 0)

    newData = []
    for item in datas:
        if item < base:
            newData.append(hidden)
        elif item < core:
            newData.append((36, 170, 131, 255))

        else:
            newData.append((173, 171, 35, 255))

    img.putdata(newData)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    enhancer = ImageEnhance.Brightness(img)

    enhanced_im = enhancer.enhance(10)
    img = enhanced_im.filter(ImageFilter.GaussianBlur(radius=1))
    return np.asarray(img)


def format_saliency(img):
    img = img.convert("RGBA")
    img.putalpha(60)
    return np.asarray(img)


def merge_img(img1, img2, index):
    img1 = img1.transpose((1, 0))
    img1 = Image.fromarray(img1, "L")
    img1.save('images/state_' + str(index) + '.jpg')
    img1 = img1.convert("RGBA")

    img2 = Image.fromarray(img2).convert("RGBA")

    img1.paste(img2, (0, 0), img2)
    img1.save('images/merge_demo_' + str(index) + '.png', "PNG")

    return np.asarray(img1)


if __name__ == '__main__':
    args = parsearg()
    env = DoomEnv(args, False)
    agent = DQNAgent(args, env.n_actions)
    weight = torch.load(args.model_savefile, map_location=lambda storage, loc: storage).to(device)
    agent.model = weight

    run()
