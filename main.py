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

# Support Vector Classification
# gamma można ustawiać inne wartości; ten parametr decyduje o tym jak bardzo będziemy trenować te modele
# większa wartość np. 0.01 daje już gorsze wyniki, przestrzeń rozmywa się na krawędzi macierzy
clf = svm.SVC(gamma=0.001)
clf2 = svm.SVC(gamma=0.01)

# #preprocessing danych
# data = digits.images.reshape(n_samples,-1)
# # data.shape
# print(data.shape)
#
# X_train,X_test,y_train,y_test = train_test_split(data,digits.target,test_size=0.5,shuffle=False)

X_train, X_test, y_train, y_test = trenuj_model(digits, 0.5, False)
# wartość 0.5 wynika z doświadczenia - przy mniejszej ilości danych lepiej dać 0.5;
# dla większej ilości danych (np. >100 000) lepiej dać 0.2 - będzie optymalny
# każdy model można przetrenowć - tzn. im dłużej trenujesz tym gorszy wynik
# dlatego te wartości trzeba dobierać doświadczalnie albo są mechanizmy dobierające optymalne wartości

clf.fit(X_train,y_train)
clf2.fit(X_train,y_train)

# predykcja na wytrenowanym modelu
predicted = clf.predict(X_test)
predicted2 = clf2.predict(X_test)

# Poniższe warto wstawić do sparametryzowanej funkcji, bo używamy 2x to samo
# zapis _, powoduje, że dana zmienna jest ignorowana podczas rzutowania (jest nam niepotrzebna, nie będzie używana)
# poniżej funkcja plt.subplots() zwraca zbiór składający się z 2 wartości i pierwszą ignoruję, biorę tylko drugą
_, axes = plt.subplots(nrows=1,ncols=4,figsize=(10,3))
for ax,image,pred in zip(axes,X_test,predicted):
    ax.set_axis_off()
    image = image.reshape(8,8)
    ax.imshow(image,cmap = plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f'predykcja: {pred}')

plt.show()

# raport klasyfikacji dla gamma=0.001
print(f"raport klasyfikacji dla klasyfikatora {clf} ->\n{metrics.classification_report(y_test,predicted)}\n")

# macierz pomyłek dla gamma=0.001
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test,predicted)
disp.figure_.suptitle(("Macierz pomyłek"))

plt.show()

# raport klasyfikacji dla gamma=0.01
print(f"raport klasyfikacji dla klasyfikatora {clf2} ->\n{metrics.classification_report(y_test,predicted2)}\n")

# macierz pomyłek dla gamma=0.01
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test,predicted2)
disp.figure_.suptitle(("Macierz pomyłek 2"))

plt.show()
