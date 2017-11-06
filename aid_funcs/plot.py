import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from functools import partial
from aid_funcs.image import im_rescale

def get_screen_size():
    """Returns the size of current screen in pixels as ndarray [width. height]"""
    from win32api import GetSystemMetrics
    scr_sz = np.array([GetSystemMetrics(0), GetSystemMetrics(1)])
    return scr_sz


def set_curr_fig_size(sz_ratio):
    """
    Set the size of the current figure to be a portion of screen size

    :param sz_ratio: scalar in range of [0,1]
    :return: None
    """

    mngr = plt.get_current_fig_manager()
    scr_sz = get_screen_size()
    margins = np.round(scr_sz * (1 - sz_ratio)/2)
    new_sz = np.round(sz_ratio * scr_sz)
    mngr.window.setGeometry(margins[0], margins[1], new_sz[0], new_sz[1])


def show_image_with_overlay(img, overlay, overlay2=np.array([]), title_str='', alpha=0.3):
    """Function for presenting image with up to 2 partially transprent binary masks as overlays

    :param img: base image array (currently grayscale only supported)
    :param overlay: binary mask to be presented as red overlay
    :param overlay2 (optional):  binary mask to be presented as blue overlay
    :param title_str (optional): string for the plot title
    :param alpha (optional, default=0.3): value of transparency of the overlay(s). should be of range [0.0,1.0]
    :return: none. plotting to new figure if none is open or to last active figure if exist
    """

    sz = np.shape(img)
    img = im_rescale(img, 0, 255)
    if np.sum(overlay) > 0:
        overlay = im_rescale(overlay, 0, 255)
    out_img = np.ndarray((sz[0], sz[1], 3))
    out_img[:, :, 0] = (1 - alpha) * img + alpha * overlay
    out_img[:, :, 1] = (1 - alpha) * img
    if overlay2.size == 0:
        out_img[:, :, 2] = (1 - alpha) * img
    else:
        if np.sum(overlay2) > 0:
            overlay2 = im_rescale(overlay2, 0, 255)
        out_img[:, :, 2] = (1 - alpha) * img + (alpha * overlay2)
    out_img = np.uint8(out_img)
    out = plt.imshow(out_img)
    plt.title(title_str)
    return out



def plot(IM,center,width,sliceNum,addsln):
    try:
        (x, y, z) = IM.shape
        val = int(sliceNum.get())+addsln
        if val > 0 and val <= z:
            sliceNum.set(val)
        center = float(center.get())
        width = float(width.get())
        sliceNum = int(sliceNum.get())

        if sliceNum <= z and sliceNum > 0 :
            plt.cla()
            plt.imshow(IM[:,:,sliceNum-1],interpolation='nearest',cmap ='gray',vmin = center-width/2,vmax = center+width/2)
            plt.title(sliceNum)
            plt.draw()

    except ValueError:
        pass



def plot3d(IM):
    IM = np.array(IM)
    (numOfSlices, xx, yy) = IM.shape
    root = Tk()
    root.title(str(numOfSlices)+' Slices')


    mainframe = ttk.Frame(root, padding="3 3 15 15")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    #scrollbar.pack(side=RIGHT, fill=Y)

    center = StringVar()
    width = StringVar()
    sliceNum = StringVar()
    center.set(-400)
    width.set(800)
    sliceNum.set(1)

    center_entry = ttk.Entry(mainframe, width=7, textvariable=center)
    center_entry.grid(column=2, row=1, sticky=(W, E))
    ttk.Label(mainframe, text="center").grid(column=2, row=2, sticky=S)

    width_entry = ttk.Entry(mainframe, width=7, textvariable=width)
    width_entry.grid(column=1, row=1, sticky=(W, E))
    ttk.Label(mainframe, text="width").grid(column=1, row=2, sticky=S)

    sliceNum_entry = ttk.Entry(mainframe, width=7, textvariable=sliceNum)
    sliceNum_entry.grid(column=3, row=1, sticky=(W, E))
    ttk.Label(mainframe, text="sliceNum").grid(column=3, row=2, sticky=S)


    #ttk.Label(mainframe, textvariable=meters).grid(column=3, row=1, sticky=(W, E))
    action_with_arg = partial(plot, IM,center,width,sliceNum,0)
    ttk.Button(mainframe, text="plot", command = action_with_arg).grid(column=3, row=3, sticky=W)

    action_with_arg = partial(plot, IM, center,width,sliceNum,1)
    ttk.Button(mainframe, text="+", command = action_with_arg).grid(column=2, row=3, sticky=W)

    action_with_arg = partial(plot, IM, center,width,sliceNum,-1)
    ttk.Button(mainframe, text="-", command = action_with_arg).grid(column=1, row=3, sticky=W)

    action_with_arg = partial(plot, IM, center, width, sliceNum, 10)
    ttk.Button(mainframe, text="++", command=action_with_arg).grid(column=2, row=4, sticky=W)

    action_with_arg = partial(plot, IM, center, width, sliceNum, -10)
    ttk.Button(mainframe, text="--", command=action_with_arg).grid(column=1, row=4, sticky=W)
    #ttk.Label(mainframe, text="meters").grid(column=3, row=2, sticky=W)

    for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

    center_entry.focus()
    root.bind('<Return>', plot)

    root.mainloop()