def analyse_data(data):

    characteristics = {
        "Nombre de lettres": [],
        "Nombre de points": [],
        "Points par lettre": [],
        "Position initiale X": [],
        "Position initiale Y": [],
        "Position finale X": [],
        "Position finale Y": [],
        "Coordonnee X minimale": [],
        "Coordonnee X maximale": [],
        "Coordonnee Y minimale": [],
        "Coordonnee Y maximale": [],
        "Largeur du trace": [],
        "Hauteur du trace": []
    }

    points_by_length = {}

    for word, coords in data:

        n_letters = len(word)
        n_points = coords.shape[1]

        x = coords[0]
        y = coords[1]

        characteristics["Nombre de lettres"].append(n_letters)
        characteristics["Nombre de points"].append(n_points)
        characteristics["Points par lettre"].append(n_points / n_letters)

        characteristics["Position initiale X"].append(x[0])
        characteristics["Position initiale Y"].append(y[0])
        characteristics["Position finale X"].append(x[-1])
        characteristics["Position finale Y"].append(y[-1])

        characteristics["Coordonnee X minimale"].append(x.min())
        characteristics["Coordonnee X maximale"].append(x.max())
        characteristics["Coordonnee Y minimale"].append(y.min())
        characteristics["Coordonnee Y maximale"].append(y.max())

        characteristics["Largeur du trace"].append(x.max() - x.min())
        characteristics["Hauteur du trace"].append(y.max() - y.min())

        # Regroupement selon le nombre de lettres
        if n_letters not in points_by_length:
            points_by_length[n_letters] = []

        points_by_length[n_letters].append(n_points)

    # Statistiques generales
    print(f"\nNombre d'echantillons : {len(data)}")

    print("\nSTATISTIQUES GENERALES")
    print(f"{'Caracteristique':30} {'Min':>12} {'Moyenne':>12} {'Max':>12}")

    for name, values in characteristics.items():

        minimum = min(values)
        maximum = max(values)
        average = sum(values) / len(values)

        print(f"{name:30} {minimum:12.2f} {average:12.2f} {maximum:12.2f}")

    # Statistiques selon le nombre de lettres
    print("\nNOMBRE DE POINTS SELON LE NOMBRE DE LETTRES")
    print(f"{'Lettres':>8} {'Quantite':>10} {'Min':>10} {'Moyenne':>10} {'Max':>10}")

    for n_letters, values in sorted(points_by_length.items()):

        minimum = min(values)
        maximum = max(values)
        average = sum(values) / len(values)

        print(f"{n_letters:8} {len(values):10} {minimum:10} {average:10.2f} {maximum:10}")