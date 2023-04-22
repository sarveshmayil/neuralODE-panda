import matplotlib.pyplot as plt
import numpngw
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import os


class GIFVisualizer(object):
    def __init__(self):
        self.frames = []

    def set_data(self, img):
        self.frames.append(img)

    def reset(self):
        self.frames = []

    def get_gif(self):
        # generate the gif
        filename = os.path.join(os.getcwd(), 'pushing_visualization.gif')
        print("Creating animated gif, please wait about 10 seconds")
        numpngw.write_apng(filename, self.frames, delay=10)
        return filename


class NotebookVisualizer(object):
    def __init__(self, fig, hfig):
        self.fig = fig
        self.hfig = hfig

    def set_data(self, img):
        plt.clf()
        plt.imshow(img)
        plt.axis('off')
        self.fig.canvas.draw()
        self.hfig.update(self.fig)

    def reset(self):
        pass


class ImageLabel(tk.Label):
    """a label that displays images, and plays them if they are gifs"""
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        self.loc = 0
        self.frames = []

        try:
            for i in count(1):
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()

    def unload(self):
        self.config(image="")
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.loc += 1
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)
