
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster

from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition
from numpy.random import RandomState
import matplotlib.pyplot as plt

def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()


if __name__ == "__main__":

    rng = RandomState(0)

    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
    n_samples, n_features = faces.shape

    # Global centering (focus on one feature, centering all samples)
    faces_centered = faces - faces.mean(axis=0)

    # Local centering (focus on one sample, centering all features)
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

    print("Dataset consists of %d faces" % n_samples)

    n_row, n_col = 2, 3
    n_components = n_row * n_col
    image_shape = (64, 64)

    plot_gallery("Faces from dataset", faces_centered[:n_components])

    pca_estimator = decomposition.PCA(
        n_components=n_components, svd_solver="randomized", whiten=True
    )
    pca_estimator.fit(faces_centered)
    plot_gallery(
        "Eigenfaces - PCA using randomized SVD", pca_estimator.components_[:n_components]
    )

    print('test')

# Q2
# Q2- SVD for face recognition. Explain how it works and show some examples.
# Bonus: Implement it on a (B&W) image data set with adequate explanation.

'''
# Flower example from class 
image = Image.open('img-flower2.jpg')
image = img_to_array(image)
image.shape

img =np.mean(image,axis=2)
img.shape

#plt.imshow(img,'gray')
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(img,'gray')
img.shape

X_centered = img - img.mean()

# calculate U, sigma, and V for SVD
U, sigma, V = np.linalg.svd(X_centered)
print(U.shape)

a=0
b=9

# reconstruct the original matrix using U, V transpose, and sigma
reconstructed= np.matrix(U[:, a:b]) * np.diag(sigma[a:b]) * np.matrix(V[a:b, :])

ax2.imshow(reconstructed,'gray')

plt.savefig('cool.png')

print('test')

'''
