# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

def edit_distance(x,y):
    lenx = len(x)
    leny = len(y)

    array = np.zeros((lenx+1,leny+1), dtype=int)

    for i in range(lenx+1):
        for j in range(leny+1):
            if i == 0:
                array[0][j] = j
            elif j == 0:
                array[i][0] = i
            elif x[i-1] == y[j-1]:
                array[i][j] = array[i-1][j-1]
            else:
                array[i][j] = min(array[i-1][j-1], array[i-1][j], array[i][j-1]) + 1

    return array[lenx][leny]

def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion
    confusion_matrix = np.zeros((26, 26))

    for i in range(len(true)):
        true_index = true[i]
        pred_index = pred[i]

        if true_index in ignore or pred_index in ignore:
            continue

        confusion_matrix[true_index][pred_index] += 1

    return confusion_matrix

if __name__ == "__main__":
    x = "allo"
    y = "pallo"
    print(edit_distance(x,y))
    x = [1,1,2,3]
    y = [2,1,2,3]
    print(confusion_matrix(x,y))