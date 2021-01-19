import numpy as np

class Augmentor(object):
    def __init__(
            self,
            use_brightness = False,
            use_darkness = False,
            use_unibright = False,
            use_unidark = False,
            use_zoom = False,
            use_colorful_dot = False,
            strength = 0.5
            ):
        self.strength = strength
        self.funcs = []
        if use_brightness:
            self.funcs.append(self.bright)
        if use_darkness:
            self.funcs.append(self.dark)
        if use_unibright:
            self.funcs.append(self.unibright)
        if use_unidark:
            self.funcs.append(self.unidark)
        if use_zoom:
            self.funcs.append(self.zoom)
        if use_colorful_dot:
            self.funcs.append(self.colorful_dot)
        self.N = len(self.funcs)

    def __call__(self, x):
        if self.strength > 0:
            N = len(self.funcs)
            r = int(np.random.rand() * N)
            y = self.funcs[r](x)
            return y
        else:
            return x

    def bright(self, img, c=0.3):
        """
        Salt and pepper noise to white.
        """
        rank = len(img.shape)
        c *= self.strength
        if rank == 3:
            h, w, _ = img.shape
            return (c*(255-img)*np.random.rand(h, w, 1)).astype(np.uint8) + img
        elif rank == 4:
            a, h, w, _ = img.shape
            return (c*(255-img)*np.random.rand(a, h, w, 1)).astype(np.uint8) + img
        elif rank == 5:
            a, b, h, w, _ = img.shape
            return (c*(255-img)*np.random.rand(a, b, h, w, 1)).astype(np.uint8) + img

    def dark(self, img, c=0.2):
        """
        Salt and pepepr noise to black.
        """
        rank = len(img.shape)
        c *= self.strength
        if rank == 3:
            h, w, _ = img.shape
            return (c*np.random.rand(h, w, 1)*img + (1-c)*img).astype(np.uint8)
        elif rank == 4:
            a, h, w, _ = img.shape
            return (c*np.random.rand(a, h, w, 1)*img + (1-c)*img).astype(np.uint8)
        elif rank == 5:
            a, b, h, w, _ = img.shape
            return (c*np.random.rand(a, b, h, w, 1)*img + (1-c)*img).astype(np.uint8)

    def unibright(self, img, c=0.6):
        """
        Uniform noise to white.
        """
        rank = len(img.shape)
        c *= self.strength
        if rank == 3 or rank == 4:
            return ((255-img)*c*np.random.rand()).astype(np.uint8) + img
        elif rank == 5:
            a = img.shape[0]
            return ((255-img)*c*np.random.rand(a, 1, 1, 1, 1)).astype(np.uint8) + img

    def unidark(self, img, c=0.5):
        """
        Uniform noise to black.
        """
        rank = len(img.shape)
        c *= self.strength
        if rank == 3 or rank == 4:
            return ((1-np.random.rand()*c)*img).astype(np.uint8)
        elif rank == 5:
            a = img.shape[0]
            return ((1-np.random.rand(a, 1, 1, 1, 1)*c)*img).astype(np.uint8)

    def zoom(img, c=1.0):
        d = (np.random.rand()*0.3+1)*(c*2) # 1~1.3
        rank = len(img.shape)
        if rank == 3:
            hb, wb, c = img.shape
            resized = cv2.resize(img, (int(wb*d), int(hb*d)))
            ha, wa, _ = resized.shape
            e1, e2 = np.random.rand(2)
            return resized[int((ha-hb)*e1):int((ha-hb)*e1+hb), int((wa-wb)*e2):int((wa-wb)*e2+wb)]
        if rank > 3:
            l = []
            for i in img:
                l.append(self.zoom(i))
            l = np.array(l)
            return l

    def colorful_dot(self, img, c=0.4):
        rank = len(img.shape)
        img = img.copy()
        if rank == 3:
            h, w, ch = img.shape
            noise = np.random.rand(h, w, ch)
            threshold = np.random.rand(h, w, ch)
            h_, w_, ch_ = np.where(threshold < c * 0.5)
            for i, j, k in zip(h_, w_, ch_):
                img[i,j,k] = (noise[i,j,k]*255).astype(np.uint8)
            return img
        if rank == 4:
            a, h, w, ch = img.shape
            noise = np.random.rand(a, h, w, ch)
            threshold = np.random.rand(a, h, w, ch)
            a_, h_, w_, ch_ = np.where(threshold < c * 0.5)
            for h, i, j, k in zip(a_, h_, w_, ch_):
                img[h, i,j,k] = (noise[h,i,j,k]*255).astype(np.uint8)
            return img
        if rank == 5:
            a, b, h, w, ch = img.shape
            noise = np.random.rand(a, b, h, w, ch)
            threshold = np.random.rand(a, b, h, w, ch)
            a_, b_, h_, w_, ch_ = np.where(threshold < c * 0.5)
            for g, h, i, j, k in zip(a_, b_, h_, w_, ch_):
                img[g,h,i,j,k] = (noise[g,h,i,j,k]*255).astype(np.uint8)
            return img

class GaussianNoiser(object):
    def __init__(self, stddev=0.005):
        self.stddev = stddev
        
    def __call__(self, x):
        rank = len(x.shape)
        if rank == 2:
            a, b = x.shape
            return x + np.random.randn(a, b) * self.stddev
        elif rank == 3:
            a, b, c = x.shape
            return x + np.random.randn(a, b, c) * self.stddev
        elif rank == 4:
            a, b, c, d = x.shape
            return x + np.random.randn(a, b, c, d) * self.stddev
        elif rank == 5:
            a, b, c, d, e = x.shape
            return x + np.random.randn(a, b, c, d, e) * self.stddev
        else:
            raise ValueError('rank of input values must be <=5.')