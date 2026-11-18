# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *
from batch_maker import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = False           # Entrainement?
    test = True                # Test?
    learning_curves = False     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # Hyperparamètres d'entraînement
    n_epochs = 150
    batch_size = 32
    learning_rate = 0.01
    train_val_split = 0.8

    # Paramètres de l'architecture
    hidden_dim = 16
    n_layers = 1
    maxlen = 5

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords("data_trainval.p")

    
    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samples = int(len(dataset) * train_val_split)
    n_val_samples = len(dataset) - n_train_samples

    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset,
        [n_train_samples, n_val_samples]
    )

    # Instanciation des dataloaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=make_batch
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=make_batch
    )


    # Instanciation du model
    model = trajectory2seq(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        device=device,
        symb2int=dataset.symb2int,
        int2symb=dataset.int2symb,
        dict_size=dataset.dict_size,
        maxlen=maxlen
    ).to(device)

    # Calcul du nombre de paramètres
    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)

    print("Device :", device)
    print("Nombre de paramètres :", n_params)

    # Initialisation des variables
    train_losses = []
    val_losses = []

    train_distances = []
    val_distances = []

    best_val_distance = float("inf")
    model_path = "baseline_model.pth"

    pad = dataset.symb2int["<pad>"]
    sos = dataset.symb2int["<sos>"]
    eos = dataset.symb2int["<eos>"]

    # Fonction de coût et optimizateur
    criterion = nn.CrossEntropyLoss(
        ignore_index=dataset.symb2int["<pad>"]
    )

    if trainning:

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            model.train()

            train_loss = 0
            train_distance = 0
            train_tokens = 0

            for inputs, targets, lengths in train_loader:

                inputs = inputs.to(device)
                targets = targets.to(device)

                # Réinitialisation des gradients
                optimizer.zero_grad()

                # Passe avant
                predictions = model(inputs, lengths)

                # Fonction de coût
                loss = criterion(predictions, targets)

                # Rétropropagation
                loss.backward()

                # Mise à jour des paramètres
                optimizer.step()

                # Accumulation de la loss
                n_tokens = (targets != pad).sum().item()

                train_loss += loss.item() * n_tokens
                train_tokens += n_tokens

                # Distance d'édition
                pred_indices = predictions.detach().argmax(dim=1).cpu().tolist()
                true_indices = targets.cpu().tolist()

                for pred, target in zip(pred_indices, true_indices):

                    if eos in pred:
                        pred = pred[:pred.index(eos)]

                    if eos in target:
                        target = target[:target.index(eos)]

                    pred = [i for i in pred if i not in (pad, sos)]
                    target = [i for i in target if i not in (pad, sos)]

                    train_distance += edit_distance(target, pred)

            train_loss /= train_tokens
            train_distance /= len(dataset_train)
            
            # Validation
            model.eval()

            val_loss = 0
            val_distance = 0
            val_tokens = 0

            with torch.no_grad():

                for inputs, targets, lengths in val_loader:

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Passe avant
                    predictions = model(inputs, lengths)

                    # Fonction de coût
                    loss = criterion(predictions, targets)

                    # Accumulation de la loss
                    n_tokens = (targets != pad).sum().item()

                    val_loss += loss.item() * n_tokens
                    val_tokens += n_tokens

                    # Distance d'édition
                    pred_indices = predictions.argmax(dim=1).cpu().tolist()
                    true_indices = targets.cpu().tolist()

                    for pred, target in zip(pred_indices, true_indices):

                        if eos in pred:
                            pred = pred[:pred.index(eos)]

                        if eos in target:
                            target = target[:target.index(eos)]

                        pred = [i for i in pred if i not in (pad, sos)]
                        target = [i for i in target if i not in (pad, sos)]

                        val_distance += edit_distance(target, pred)

            val_loss /= val_tokens
            val_distance /= len(dataset_val)

            # Ajouter les loss aux listes
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_distances.append(train_distance)
            val_distances.append(val_distance)

            print(f"Epoch {epoch}/{n_epochs}")

            print(f"Training loss: {train_loss:.6f}, "
                  f"Edit Distance: {train_distance:.6f}")

            print(f"Validation loss: {val_loss:.6f}, "
                  f"Edit Distance: {val_distance:.6f}")

            # Enregistrer les poids
            if val_distance < best_val_distance:
                best_val_distance = val_distance

                torch.save(model.state_dict(), model_path)

                print("Meilleur modèle enregistré")


            # Affichage
            if learning_curves:
                # visualization
                if learning_curves and epoch == n_epochs:
                    epochs = range(1, n_epochs + 1)

                    # Fonction de coût
                    plt.figure()
                    plt.plot(epochs, train_losses, label="Training")
                    plt.plot(epochs, val_losses, label="Validation")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title("Learning curves - Loss")
                    plt.legend()
                    plt.grid()

                    # Distance d'édition
                    plt.figure()
                    plt.plot(epochs, train_distances, label="Training")
                    plt.plot(epochs, val_distances, label="Validation")
                    plt.xlabel("Epoch")
                    plt.ylabel("Mean edit distance")
                    plt.title("Learning curves - Edit distance")
                    plt.legend()
                    plt.grid()

                    plt.show()

    if test:
        # Évaluation
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

        model.eval()

        # Charger les données de tests
        dataset_test = HandwrittenWords("data_test.p")

        test_loader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=make_batch
        )

        # Initialisation des résultats de test
        test_loss = 0
        test_distance = 0
        test_tokens = 0

        test_confusion = np.zeros((26, 26))

        with torch.no_grad():

            for inputs, targets, lengths in test_loader:

                inputs = inputs.to(device)
                targets = targets.to(device)

                # Passe avant
                predictions = model(inputs, lengths)

                # Fonction de coût
                loss = criterion(predictions, targets)

                n_tokens = (targets != pad).sum().item()

                test_loss += loss.item() * n_tokens
                test_tokens += n_tokens

                # Extraction des symboles prédits
                pred_indices = predictions.argmax(dim=1).cpu().tolist()
                true_indices = targets.cpu().tolist()

                for pred, target in zip(pred_indices, true_indices):

                    # Matrice de confusion : ignorer les positions après EOS
                    if eos in pred:
                        stop = pred.index(eos)
                        pred[stop:] = [pad] * (len(pred) - stop)

                    test_confusion += confusion_matrix(
                        target,
                        pred,
                        ignore=[pad, sos, eos]
                    )

                    # Distance d'édition
                    if eos in target:
                        target = target[:target.index(eos)]

                    pred = [i for i in pred if i not in (pad, sos)]
                    target = [i for i in target if i not in (pad, sos)]

                    test_distance += edit_distance(target, pred)

        # Calcul des moyennes
        test_loss /= test_tokens
        test_distance /= len(dataset_test)

        # Affichage de l'attention
        if gen_test_images:

            for i in range(3):

                # Sélection d'un échantillon aléatoire
                idx = np.random.randint(0, len(dataset_test))

                inputs, targets, lengths = make_batch(
                    [dataset_test[idx]]
                )

                # Passe avant
                with torch.no_grad():
                    predictions = model(inputs.to(device), lengths)

                # Récupération des résultats
                pred = predictions.argmax(dim=1)[0].cpu().tolist()
                target = targets[0].tolist()

                attention = model.attention_weights[0].cpu().numpy()
                coords = inputs[0].numpy()

                # Représentation des symboles sous forme de texte
                pred_word = " ".join(dataset.int2symb[t] for t in pred)
                target_word = " ".join(dataset.int2symb[t] for t in target)

                # Création des graphiques
                fig, axes = plt.subplots(
                    maxlen, 1,
                    figsize=(10, 7),
                    sharex=True,
                    sharey=True
                )

                fig.suptitle(
                    f"Target: {target_word}\nPrediction: {pred_word}"
                )

                for j in range(maxlen):
                    # Tracé complet en gris
                    axes[j].plot(
                        coords[0],
                        coords[1],
                        color="gray"
                    )

                    # Sélection des points ayant une attention importante
                    mask = attention[j] >= 0.2 * attention[j].max()

                    # Affichage des points sélectionnés
                    axes[j].scatter(
                        coords[0][mask],
                        coords[1][mask],
                        color="black",
                        s=10
                    )

                    axes[j].set_ylabel(dataset.int2symb[pred[j]])

                axes[-1].set_xlabel("X")

                plt.tight_layout()

        # Affichage des résultats de test
        print("\nTest results")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Edit Distance: {test_distance:.6f}")
        
        # Affichage de la matrice de confusion
        letters = list("abcdefghijklmnopqrstuvwxyz")

        plt.figure(figsize=(10, 10))

        plt.imshow(test_confusion, cmap="Blues")
        plt.colorbar()

        plt.xticks(range(26), letters)
        plt.yticks(range(26), letters)

        plt.xlabel("Predicted letter")
        plt.ylabel("True letter")
        plt.title("Confusion matrix")

        plt.show()