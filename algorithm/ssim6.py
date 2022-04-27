from PIL import Image
import numpy as np


class SSIM:
    def __init__(self, alg, samplePath=None, testPath=None):
        self.samplePath = samplePath
        self.testPath = testPath
        self.algName = alg

    def similarity(self):
        im1= Image.open(self.samplePath).convert('RGB')
        getRGB = im1.size
        img2 = Image.open(self.testPath).convert('RGB')
        im2 = img2.resize((getRGB[0], getRGB[1]), Image.BILINEAR)
        im1 = np.array(im1)
        im2 = np.array(im2)
        mu1 = im1.mean()
        mu2 = im2.mean()
        sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
        sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
        sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
        k1, k2, L = 0.01, 0.03, 255
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        C3 = C2 / 2
        l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
        c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
        s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
        ssim = l12 * c12 * s12
        if ssim>0.5:
            return 1
        return 0
        # return ssim


