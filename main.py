import matplotlib.pyplot as plt
from sklearn import datasets,metrics,svm
from sklearn.model_selection import train_test_split
from funkcje.funkcje_svm import laduj_dane
from funkcje.funkcje_svm import trenuj_model

# digits = datasets.load_digits()
#
# _, axes = plt.subplots(nrows=1,ncols=4,figsize=(10,3))
# for ax,image,label in zip(axes,digits.images,digits.target):
#     ax.set_axis_off()
#     ax.imshow(image,cmap = plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f'trening: {label}')
#
# plt.show()

digits = laduj_dane(datasets.load_digits(),1,4,10,3)

# n_samples = len(digits.images)
# # n_samples
# print(n_samples)

clf = svm.SVC(gamma=0.001)

# #preprocessing danych
# data = digits.images.reshape(n_samples,-1)
# # data.shape
# print(data.shape)
#
# X_train,X_test,y_train,y_test = train_test_split(data,digits.target,test_size=0.5,shuffle=False)

X_train, X_test, y_train, y_test = trenuj_model(digits, 0.5, False)

clf.fit(X_train,y_train)

# predykcja na wytrenowanym modelu
predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1,ncols=4,figsize=(10,3))
for ax,image,pred in zip(axes,X_test,predicted):
    ax.set_axis_off()
    image = image.reshape(8,8)
    ax.imshow(image,cmap = plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f'predykcja: {pred}')

plt.show()

print(f"raport klasyfikacji dla klasyfikatora {clf} ->\n{metrics.classification_report(y_test,predicted)}\n")

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test,predicted)
disp.figure_.suptitle(("Macierz pomyłek"))

plt.show()
