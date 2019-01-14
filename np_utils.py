import numpy as np
# from skimage import color
# from sklearn.cluster import KMeans

def normalize(x, p=False):

    if p:
        print ("normalize ration of the perturbation:", 1./(x.max()-x.min()))

    return (x-x.min())/(x.max()-x.min())


def compute_amp(w, x, xp, r):

    w_ = np.reshape(w, [-1])
    x_ = np.reshape(x, [-1])
    xp_ = np.reshape(xp, [-1])
    r_ = np.reshape(r, [-1])

    return np.dot(w_, xp_ - x_)/np.dot(w_, r_)

def distanceFromHyperplane(x0, w, xp):
    x0_ = x0.reshape([-1])
    w_ = w.reshape([-1])
    xp_ = xp.reshape([-1])

    return np.dot(w_, x0_-xp_)


# def cluster_img(img, n_clusters, transpose=True):
#     if len(img.shape) == 4:
#         img1 = np.copy(img[0, :, :, :].astype(np.uint8))
#     else:
#         img1 = np.copy(img.astype(np.uint8))
#
#     if transpose:
#         w = img1.shape[1]
#         h = img1.shape[2]
#         img1 = np.transpose(img1, [1, 2, 0])
#     else:
#         w = img1.shape[0]
#         h = img1.shape[1]
#
#     lab = color.rgb2lab(img1)
#     lab_v = np.reshape(lab, [w * h, 3])
#
#     clt = KMeans(n_clusters=n_clusters)
#     clt.fit(lab_v)
#     clcs = clt.labels_
#     clcs = np.reshape(clcs, [w, h])
#
#     return clcs

def inv_tf(x, mean, std):

    """
    taken from the sparsefool implementation at https://github.com/LTS4/SparseFool
    """
    for i in range(len(mean)):

        x[i] = np.multiply(x[i], std[i], dtype=np.float32)
        x[i] = np.add(x[i], mean[i], dtype=np.float32)

    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)

    return x
