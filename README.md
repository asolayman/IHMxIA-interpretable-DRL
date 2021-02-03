# IHMxIA-interpretable-DRL
> TP IHMxIA: Interpretable Deep Reinforcement Learning

**Réalisé par Alexandre DEVILLERS p1608591 & Solayman AYOUBI p1608583**

## Installation

```sh
pip3 install -r requirements.txt
```

## Utilisation

```sh
python3 run.py
```

## Visualisation

### Saliency Maps

Cette [vidéo](https://github.com/asolayman/IHMxIA-interpretable-DRL/blob/main/video/video_30.mp4) montre sur 30 épisodes la carte de saillance, qui est représentée par le halo turquoise et jaune. On remarque que le réseau de neurones porte intérêt sur le monstre à tuer.

<img src="https://github.com/asolayman/IHMxIA-interpretable-DRL/blob/main/video/video_30.gif" alt="drawing" width="1120" height="640"/>

### Projection Umap

On constate que les projections séparent les données en fonction de l'action à réaliser.

Paramètres : `n_neighbors=25, min_dist=0.7`
![Projection 1](https://github.com/asolayman/IHMxIA-interpretable-DRL/blob/main/projection1.png)

Paramètres : `n_neighbors=7, min_dist=0.4`
![Projection 2](https://github.com/asolayman/IHMxIA-interpretable-DRL/blob/main/projection2.png)
