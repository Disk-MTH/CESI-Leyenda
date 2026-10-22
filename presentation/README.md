# Présentation de soutenance

Support de soutenance du projet Leyenda (TouNum) : 15 slides et 4 annexes, au format 16:9, pour environ 17 minutes
de présentation et 2 minutes de démo.

La présentation se joue depuis Claude :
**https://claude.ai/artifact/VJUaRKo9u7eGptki94H4fG**

Ce dossier en garde une copie versionnée : le texte de chaque slide, les notes orateur et les images.

## Lancer la présentation le jour de la soutenance

1. Ouvrir le lien ci-dessus dans Chrome ou Edge, connecté au compte Claude propriétaire de la présentation (ou à un
   compte avec qui elle a été partagée via le menu Partager).
2. Cliquer sur **Présenter** : la présentation passe en plein écran. `F11` si le navigateur garde ses barres.
3. Avancer avec `→`, `Espace` ou un clic (les animations « au clic » se déclenchent une à une), reculer avec `←`.
   `Échap` quitte le mode présentation.
4. Le projecteur doit être en mode **dupliqué** ou la fenêtre placée sur l’écran du projecteur.
5. Démo (slide 15) : basculer sur `notebooks/livrable3_captioning.ipynb`, dernière cellule, ou en terminal :
   `uv run python scripts/caption_folder.py <dossier> --out captions.csv`.

### Avant de commencer

- Tester le lien sur la machine et le réseau de la salle (la présentation est en ligne).
- **Plan B hors ligne** : depuis la présentation, menu Partager › Exporter, télécharger un **PDF** et un **PPTX**
  et les garder sur une clé USB. L’animation de la slide 3 y apparaît figée ; tout le reste est identique.
- Lancer une fois la cellule de démo à vide pour que le GPU soit prêt, et avoir la vidéo de secours ouverte.

## Répartition

| Slides | Partie | Orateur |
|---|---|---|
| 1 à 6 | Introduction, module 1 : trier | Matthis |
| 7 à 10 | Module 2 : nettoyer | Julien |
| 11 à 15 | Module 3 : décrire, workflow complet, démo | Mathieu |
| 16 à 19 | Annexes, pour les questions du jury | selon la question |

## Contenu du dossier

| Chemin | Contenu |
|---|---|
| `deck.json` | ordre des slides, sections, polices |
| `slides/*.html` | une slide par fichier ; les notes orateur sont dans la balise `<aside>` à la fin de chaque fichier |
| `assets/*.png` | images utilisées par les slides, extraites des notebooks exécutés |

Les fichiers `slides/*.html` sont au format de l’éditeur de slides de Claude (balises `x-embed`, `x-connector`…) :
ils ne s’ouvrent pas tels quels dans un navigateur. Pour modifier la présentation, passer par le lien ci-dessus,
puis mettre à jour ce dossier.
