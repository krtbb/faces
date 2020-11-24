import matplotlib.pyplot as plt
import numpy as np

def draw_subplot(
        x, # (time, motor_num)
        y, # (time, motor_num)
        n, 
        yscale = None, 
        linewidth = 1,
        title = None, 
        dashed = None, 
        ylim = [None, None],
        max_iter = None
        ):
    plt.subplot(n[0], n[1], n[2])
    if max_iter is None:
        if x is None:
            plt.plot(y, linewidth=linewidth)
        else:
            plt.plot(x, y, linewidth=linewidth)
    else:
        if x is None:
            for i in range(max_iter):
                plt.plot(y[:,i], linewidth=linewidth, c=get_colorcode(i))
        else:
            for i in range(max_iter):
                plt.plot(x, y[:, i], linewidth=linewidth, c=get_colorcode(i))
    
    if dashed is not None:
        if max_iter is None:
            if x is None:
                plt.plot(dashed, linestyle='dashed', linewidth=linewidth)
            else:
                plt.plot(x, dashed, linestyle='dashed', linewidth=linewidth)
        else:
            if x is None:
                for i in range(max_iter):
                    plt.plot(dashed[:, i], linestyle='dashed', linewidth=linewidth, c=get_colorcode(i))
            else:
                for i in range(max_iter):
                    plt.plot(x, dashed[:, i], linestyle='dashed', linewidth=linewidth, c=get_colorcode(i))

    if title:
        plt.title(title)
    if yscale:
        plt.yscale(yscale)
    plt.ylim(ylim[0], ylim[1])
    plt.grid()

def draw_images(data, n, title=None):
    # data.shape must be "[seq, h, w, c]", rgb style
    seq, h, w, c = data.shape
    plt.subplot(n[0], n[1], n[2])
    whole_img = np.zeros((h*seq//10, w*10, c))
    for i in range(seq//10):
        row_img = np.zeros((h, w*10, c))
        for j in range(10):
            row_img[:,j*w:(j+1)*w,:] = data[i*10+j, :, :, :]
        whole_img[i*h:(i+1)*h, :, :] = row_img
    whole_img = whole_img.astype(np.uint8)
    plt.subplot(n[0], n[1], n[2])
    plt.imshow(whole_img)

    #from IPython import embed; embed()
    #sys.exit
    if title:
        plt.title(title)

def get_colorcode(i, style='matplotlib'):
    if style == 'matplotlib':
        l = [ "#1f77b4", # blue
                "#ff7f0e", # orange
                "#2ca02c", # green
                "#d62728", # red
                "#9467bd", # purple
                "#8c564b", # brown
                "#e377c2", # pink
                "#7f7f7f", # gray
                "#bcbd22", # citrus
                "#17becf" ] # right blue
    else:
        raise NotImplementedError('style != {}'.format(style))

    return l[i % len(l)]