import matplotlib.pyplot as plt
from sklearn import datasets,metrics,svm
from sklearn.model_selection import train_test_split


def laduj_dane(ds_digits, nb_rows, nb_cols, fig_hgt, fig_wdt):

    _, axes = plt.subplots(nrows=nb_rows, ncols=nb_cols, figsize=(fig_hgt, fig_wdt))
    for ax, image, label in zip(axes, ds_digits.images, ds_digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f'trening: {label}')

    plt.show()

    return ds_digits

