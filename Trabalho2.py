import pandas as pd

import sys

import numpy as np


def kernel_instancia(i):
    return [i[0] ** 2, i[0] * i[1] * np.sqrt(2), i[1] ** 2]


def kernel(db):
    df = pd.read_csv(db, delim_whitespace=True)
    ds = df.iloc[:, 1: -1]
    ds = ds.to_numpy()
    dsR3 = list(map(kernel_instancia, ds))
    print(*dsR3, sep='\n')


def classificador(hiperplano, ponto):
    x = ponto[0]
    y = ponto[1]

    r = eval(hiperplano)
    if r >= 0:
        print('Classe positiva')
    else:
        print('Classe negativa')


def distancia_euclideana(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


def indutor(db):
    df = pd.read_csv(db, delim_whitespace=True)
    positivas = df[df.iloc[:, -1] == '+']
    positivas = positivas.to_numpy()
    negativas = df[df.iloc[:, -1] == '-']
    negativas = negativas.to_numpy()

    # vetor de todas as distâncias
    distancias = []
    # vetor das distâncias ponto por ponto
    pontoaponto = []

    for p in positivas:
        for n in negativas:
            pontoaponto.append(p[0])
            pontoaponto.append(n[0])
            pontoaponto.append(distancia_euclideana(p[1:3], n[1:3]))
            distancias.append(pontoaponto)
            pontoaponto = []

    distancias = np.array(distancias)

    # os menores valores de distância
    arg_dist = distancias[:, -1]
    arg_dist = list(map(np.float, arg_dist))
    arg_dist = np.argsort(arg_dist)

    # as primeiras instâncias mais proximas
    print(f'dist({distancias[arg_dist[0]][0]}, {distancias[arg_dist[0]][1]})',
          '=', f'{distancias[arg_dist[0]][-1]}')
    # as segundas instâncias mais proximas
    print(f'dist({distancias[arg_dist[1]][0]}, {distancias[arg_dist[1]][1]})',
          '=', f'{distancias[arg_dist[1]][-1]}')

    # pivots
    p1 = distancias[arg_dist[0]][0]
    p2 = distancias[arg_dist[1]][0]
    n1 = distancias[arg_dist[0]][1]
    print(f'suportes escolhidos: ', p1, ',', p2, ',', n1)

    # buscando os valores por id
    p1 = np.asarray(df[df.id == p1].iloc[:, 1:3])[0]
    p2 = np.asarray(df[df.id == p2].iloc[:, 1:3])[0]
    n1 = np.asarray(df[df.id == n1].iloc[:, 1:3])[0]

    # variáveis
    x1 = p1[0]
    x2 = p1[1]
    x3 = p2[0]
    x4 = p2[1]
    x5 = n1[0]
    x6 = n1[1]

    w1 = (-2 * (x4 - x2)) / (((x4 - x2) * (x5 - x1) - (x3 - x1) * (x6 - x2)))

    w2 = (-2 / (x6 - x2)) + (2 * (x4 - x2) * (x5 - x1) * (x5 - x1)) / (((x4 - x2) *
                (x5 - x1) * (x6 - x2)) - ((x3 - x1) * (x6 - x2) * (x6 - x2)))

    b = 1 - w1 * x1 - w2 * x2

    print(f'hiperplano: {w1:.3f}x + {w2:.3f}y + {b:.3f} = 0')


def main():
    parametros = sys.argv

    modo = parametros[1]

    if modo == '--classificador':
        hiperplano = parametros[2]
        ponto = list(map(np.float, parametros[3].strip(')(').split(',')))
        classificador(hiperplano, ponto)

    else:
        db = parametros[2]
        if modo == '--kernel':
            kernel(db)
        elif modo == '--indutor':
            indutor(db)


if __name__ == "__main__":
    main()
