import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

from analyse_data import analyse_data

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        # Extraction des symboles
        self.symb2int = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7,
            'i': 8,
            'j': 9,
            'k': 10,
            'l': 11,
            'm': 12,
            'n': 13,
            'o': 14,
            'p': 15,
            'q': 16,
            'r': 17,
            's': 18,
            't': 19,
            'u': 20,
            'v': 21,
            'w': 22,
            'x': 23,
            'y': 24,
            'z': 25,
            pad_symbol: 26,
            start_symbol: 27,
            stop_symbol: 28,
        }
        self.int2symb = {
            0: 'a',
            1: 'b',
            2: 'c',
            3: 'd',
            4: 'e',
            5: 'f',
            6: 'g',
            7: 'h',
            8: 'i',
            9: 'j',
            10: 'k',
            11: 'l',
            12: 'm',
            13: 'n',
            14: 'o',
            15: 'p',
            16: 'q',
            17: 'r',
            18: 's',
            19: 't',
            20: 'u',
            21: 'v',
            22: 'w',
            23: 'x',
            24: 'y',
            25: 'z',
            26: pad_symbol,
            27: start_symbol,
            28: stop_symbol,
        }

        self.dict_size = 29
        
        # Ajout du padding aux séquences
        self.targets = []

        for word, coords in self.data:
            letters = list(word)
            if len(letters) < 5:
                letters.append(stop_symbol)
            while len(letters) < 5:
                letters.append(pad_symbol)

            tokens = []
            for letter in letters:
                tokens.append(self.symb2int[letter])

            self.targets.append(
                torch.tensor(tokens, dtype=torch.long)
            )
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        RNN_input = torch.tensor(
            self.data[idx][1],
            dtype=torch.float32
        )

        RNN_input = RNN_input / 1000.0

        target = self.targets[idx]

        return RNN_input, target

    def visualisation(self, idx):
        # Visualisation des échantillons
        word = self.data[idx][0]
        coords = self.data[idx][1]

        plt.figure()
        plt.plot(coords[0], coords[1])
        plt.title(word)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.show()

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    #a = HandwrittenWords('data_trainval.p')
    a = HandwrittenWords('data_test.p')
    analyse_data(a.data)

    RNN_input, target = a[0]
    print("Mot original :", a.data[0][0])
    print("Dimension entrée :", RNN_input.shape)
    print("Type entrée :", RNN_input.dtype)
    print("Cible :", target)
    print("Dimension cible :", target.shape)
    print("Type cible :", target.dtype)

    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))