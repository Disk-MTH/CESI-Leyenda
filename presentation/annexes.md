# Fiche d’oral des annexes (slides 16 à 23)

Les annexes ne se présentent pas d’office : on y va quand le jury pose une question. Chaque annexe répond à **une
question**, écrite en titre, et se termine par une phrase **« À retenir »** : si tu n’as que dix secondes, dis cette
phrase.

Pour chaque annexe, cette fiche donne :

- **L’idée** : ce qu’il faut avoir compris, en deux phrases ;
- **Le script** : ce qu’on dit en montrant la slide (environ une minute ; le même texte est dans les notes orateur) ;
- **Les détails** : les chiffres et les choix techniques à connaître ;
- **Les subtilités** : les points où on peut se faire piéger, et comment les expliquer ;
- **Questions probables** : ce que le jury peut demander, avec une réponse courte.

Tous les chiffres viennent des notebooks exécutés en entier sur GPU, graine 42. D’une exécution à l’autre, les
dernières décimales peuvent bouger un peu (opérations GPU parallèles).

| Annexe | Slide | Question | Module |
|---|---|---|---|
| A1 | 16 | Comment notre CNN décide « photo » ? | Trier |
| A2 | 17 | Notre réseau est-il trop simple ou trop gros ? | Trier |
| A3 | 18 | Qu’est-ce qui a amélioré notre réseau ? | Trier |
| A4 | 19 | À partir de quelle probabilité dit-on « photo » ? | Trier |
| A5 | 20 | Trois débruiteurs : qu’est-ce qui change ? | Nettoyer |
| A6 | 21 | Quand faut-il débruiter une photo ? | Nettoyer |
| A7 | 22 | Le modèle de légende, pièce par pièce | Décrire |
| A8 | 23 | Comment note-t-on une légende ? | Décrire |

---

## A1 · Comment notre CNN décide « photo » ? (slide 16)

### L’idée
Un réseau convolutif passe de petits filtres appris sur l’image. À chaque bloc, l’image rétrécit de moitié pendant
que le nombre de « cartes » (une par motif détecté) double ; à la fin, une moyenne par carte résume l’image en 128
nombres, et deux petites couches donnent la probabilité que ce soit une photo.

### Le script
« On lit le schéma de gauche à droite. À gauche, l’image, réduite en 128 par 128 pixels, avec ses trois couleurs.
Chaque flèche bleue est un bloc : on passe des petits filtres 3 par 3 sur toute l’image, on normalise, on garde les
valeurs positives, puis on divise la taille par deux. Après quatre blocs, l’image ne fait plus que 8 par 8, mais on a
128 cartes, chacune spécialisée dans un motif : un bord, une texture, une forme. On fait la moyenne de chaque carte :
il reste 128 nombres qui résument l’image. Une couche de 64 neurones les combine, et la sortie donne la probabilité
que ce soit une photo. Pour apprendre, la perte est l’entropie croisée binaire : elle punit surtout les erreurs
commises avec assurance. L’optimiseur est Adam. Le réseau n’a que 106 000 paramètres, parce que la moyenne par carte
évite une énorme couche dense. »

### Les détails
- **Un bloc** = `Conv2D 3×3` → `BatchNormalization` → `ReLU` → `MaxPooling2D 2×2`. Filtres : 16, 32, 64, 128.
- **Tailles** : 128×128×3 → 64×64×16 → 32×32×32 → 16×16×64 → 8×8×128 → moyenne globale (128) → Dense 64 + ReLU →
  Dense 1 + sigmoïde.
- **Paramètres** : 106 481 au total, dont 480 non entraînables (moyennes et variances glissantes de BatchNorm). Par
  couche : conv1 432, conv2 4 608, conv3 18 432, conv4 73 728, dense 8 256, sortie 65, plus les BatchNorm.
- **Perte** : entropie croisée binaire, `L = −[y·log p + (1 − y)·log(1 − p)]`. Elle est faible quand le réseau a
  raison et en est sûr, très forte quand il se trompe avec assurance (une peinture classée photo à 99 %).
- **Optimiseur** : Adam, pas 10⁻³ (β₁ = 0,9, β₂ = 0,999), paquets (batchs) de 64 images.
- **Données** : 12 000 images de modélisation (6 000 photos, 6 000 non-photos dont 2 800 peintures, la classe la plus
  difficile), découpées 70 / 15 / 15 avec stratification par **type d’image** (pas seulement photo / non-photo). Plus
  un jeu de test « archive réaliste » à part : 4 605 images, 22 % de photos.
- La mise à l’échelle des pixels (÷ 255) est **dans** le modèle : en production, on lui donne directement des images
  brutes 128×128.

### Les subtilités
- **Les convolutions n’ont pas de biais** : 3 × 3 × 3 × 16 = 432 exactement. C’est voulu, BatchNorm juste après a son
  propre décalage, un biais ferait double emploi.
- **Pourquoi la moyenne globale** (`GlobalAveragePooling`) : une couche dense branchée sur 8 × 8 × 128 = 8 192 valeurs
  demanderait 8 192 × 64 ≈ 524 000 poids rien que pour elle, cinq fois tout notre réseau. Avec la moyenne, il en faut
  8 256. Moins de paramètres, moins de risque d’apprendre par cœur.
- **Redimensionnement sans garder les proportions** : une page A4 de texte est écrasée en carré. Ce n’est pas grave
  pour une décision photo / pas photo, mais ça fait perdre les détails fins qui aident à distinguer une peinture
  réaliste d’une photo.
- **Le modèle en production n’est pas ce CNN** : c’est MobileNetV2 (slide 6). Notre CNN reste le modèle « maison »
  étudié en détail, plus léger et sans poids pré-entraînés.

### Questions probables
- *Pourquoi un CNN et pas un réseau dense ?* Le même filtre est appliqué partout dans l’image : on exploite sa
  structure 2D avec très peu de paramètres. Un réseau dense sur 128 × 128 × 3 pixels en aurait des millions.
- *À quoi sert BatchNorm ?* Elle normalise les activations de chaque paquet : l’apprentissage est plus rapide et plus
  stable. Petit effet régularisant en plus.
- *Pourquoi ReLU ?* C’est la non-linéarité `max(0, x)` : simple, rapide, et elle évite que les gradients s’éteignent.
- *Pourquoi 128 × 128 ?* Compromis : assez pour voir textures et formes, assez petit pour entraîner une quinzaine de
  réseaux dans le notebook.
- *Comment avez-vous procédé ?* Par difficulté croissante, comme le conseille l’énoncé : photo contre texte (accuracy
  équilibrée 1,000), puis on ajoute schémas, dessins, et enfin peintures (0,893). Les peintures concentrent les
  erreurs.

---

## A2 · Notre réseau est-il trop simple ou trop gros ? (slide 17)

### L’idée
On a mesuré le **biais** (trop simple pour apprendre) et la **variance** (apprend par cœur) en faisant varier la
taille du réseau et la quantité d’images. Notre taille de référence suffit ; son seul défaut est un léger
sur-apprentissage, qu’on traite par régularisation (A3).

### Le script
« Deux erreurs possibles. Le biais : le réseau est trop simple, il n’arrive même pas à bien apprendre ses images
d’entraînement. La variance : il apprend par cœur, très bon à l’entraînement, moins bon sur des images nouvelles.
Pour savoir où on en est, on a entraîné trois tailles de réseau sur les mêmes images. Le petit, 485 paramètres,
plafonne à 77 % en validation et 80 % en entraînement : c’est du biais. Notre réseau de référence fait 91 %. Le
gros, 4,3 fois plus de paramètres, fait 90 % : grossir n’apporte rien, la taille n’est pas notre limite. À droite :
avec 840 images, l’écart entre entraînement et validation est de 5 points ; avec 8 400, il tombe à 2 points. Un
écart qui se referme quand on ajoute des données, c’est de la variance : plus d’images ou de la régularisation
aident. C’est l’annexe suivante. »

### Les détails

| Modèle | Paramètres | Entraînement | Validation | Écart | Diagnostic du notebook |
|---|---|---|---|---|---|
| Trop simple | 485 | 0,795 | 0,766 | 0,030 | biais |
| Référence | 106 481 | 0,929 | 0,908 | 0,021 | variance |
| Trop complexe | 455 905 | 0,932 | 0,902 | 0,030 | équilibré |

Courbe d’apprentissage (modèle de référence, moyenne des 3 dernières époques) :

| Images d’entraînement | 840 | 2 100 | 4 200 | 8 400 |
|---|---|---|---|---|
| Entraînement | 0,871 | 0,908 | 0,921 | 0,927 |
| Validation | 0,819 | 0,848 | 0,806 | 0,905 |

### Les subtilités
- **Pourquoi la référence est classée « sur-apprentissage » avec un écart de 2 points seulement ?** Le diagnostic
  automatique a des seuils fixes : biais si l’accuracy d’entraînement est sous 0,90 ; variance si l’écart
  entraînement − validation dépasse 0,05, **ou** si la perte de validation finale est plus de 0,05 au-dessus de son
  minimum. La référence est classée variance par ce second critère : sa perte de validation finale (0,248) est à
  0,056 au-dessus de son meilleur point. C’est un cas limite, à 0,006 du seuil. Sur la slide, on le dit avec des
  mots : « la perte de validation remonte en fin d’entraînement ».
- **Le gros réseau « équilibré » n’est pas meilleur** : il passe juste sous le même seuil, mais sa validation est plus
  basse (0,902 contre 0,908). Ne pas se laisser enfermer dans l’étiquette : le chiffre qui compte, c’est la
  validation.
- **Le creux à 4 200 images** (0,806) n’est pas une tendance : la validation d’une seule époque peut bouger de
  plusieurs points (BatchNorm, paquets aléatoires). On a déjà moyenné les 3 dernières époques pour limiter ce bruit.
  Ce qui compte, c’est l’écart qui passe de 0,052 à 0,022.
- **Biais et variance se lisent ensemble** : le petit réseau a un petit écart (0,03) mais une accuracy basse partout,
  c’est du biais ; plus de données ne le corrigerait pas, il lui faut plus de capacité.

### Questions probables
- *Comment corriger le biais ?* Plus de capacité (filtres, couches) ou plus d’époques. Plus de données n’aide pas.
- *Comment corriger la variance ?* Plus de données, augmentation, dropout, L2, early stopping, ou un réseau plus petit.
- *Pourquoi ne pas avoir pris le gros réseau ?* Il est 4,3 fois plus lourd pour 0,6 point de moins en validation.

---

## A3 · Qu’est-ce qui a amélioré notre réseau ? (slide 18)

### L’idée
On a testé quatre protections contre le sur-apprentissage **une par une**, puis toutes ensemble, et on les a jugées
sur la validation. L’early stopping gagne ; tout combiner fait pire, parce que trop de freins font basculer vers le
biais.

### Le script
« Notre réseau de référence sur-apprenait un peu. On a essayé quatre protections classiques, une à la fois. Avec
l’augmentation, l’image est retournée, tournée ou zoomée à chaque passage, donc le réseau ne revoit jamais exactement
la même. Le dropout éteint au hasard 40 % des neurones à chaque pas, pour qu’aucun ne devienne indispensable. Le L2
pénalise les poids trop grands. L’early stopping surveille la validation et s’arrête quand elle ne progresse plus, en
gardant le meilleur état. On compare la perte de validation : plus la barre est courte, mieux c’est. L’early
stopping gagne, avec la perte la plus basse, 0,209, et la meilleure accuracy, 92 %. Tout combiner fait moins bien :
trop de freins, le réseau n’apprend plus assez. »

### Les détails

| Configuration | Époques (retenue) | Entraînement | Perte val. | Accuracy val. | Diagnostic |
|---|---|---|---|---|---|
| Référence | 20 (20) | 0,929 | 0,248 | 0,908 | variance |
| + augmentation | 20 (20) | 0,904 | 0,232 | 0,896 | équilibré |
| + dropout 0,4 | 20 (20) | 0,917 | 0,229 | 0,906 | équilibré |
| + L2 10⁻⁴ | 20 (20) | 0,926 | 0,294 | 0,884 | variance |
| **+ early stopping** | 24 (19) | 0,930 | **0,209** | **0,920** | équilibré |
| Tout combiné | 12 (7) | 0,861 | 0,315 | 0,877 | biais |

- **Augmentation** : miroir horizontal, rotation jusqu’à ±18° (5 % d’un tour), zoom ±10 %, contraste ±10 %, seulement
  pendant l’entraînement.
- **Early stopping** : patience 5 époques sur la perte de validation, meilleurs poids restaurés.
- Le choix se fait sur la **perte de validation** ; les jeux de test ne servent qu’après, pour la mesure finale.
- Modèle retenu : 0,930 en entraînement, 0,920 en validation, diagnostic équilibré.

### Les subtilités
- **Accuracy ou précision ?** Ici on parle d’accuracy : la part d’images bien classées. En A4, la « précision » a son
  sens technique : la part de vraies photos parmi les images acceptées. Ne pas mélanger les deux à l’oral.
- **Pourquoi juger sur la perte et pas sur l’accuracy ?** La perte tient compte de la confiance du réseau, l’accuracy
  à 0,5 non ; la perte est aussi plus stable d’une époque à l’autre.
- **L’augmentation baisse la perte mais aussi un peu l’accuracy** : elle rend l’entraînement plus difficile. Avec
  plus d’époques, elle aurait peut-être fini mieux.
- **Le L2 fait pire** : la valeur 10⁻⁴ n’a pas été réglée, on l’a testée telle quelle. On ne conclut pas que le L2
  est mauvais en général.
- **Budgets inégaux** : les configurations avec early stopping avaient droit à 25 époques au lieu de 20, puisqu’elles
  décident elles-mêmes quand s’arrêter. Celle retenue s’est arrêtée à 24 et a gardé l’époque 19. C’est dit dans le
  notebook ; à reconnaître si on nous le fait remarquer.
- **« Tout combiné » s’arrête trop tôt** (époque 7 retenue, 86 % en entraînement) : chaque frein réduit ce que le
  réseau apprend, cumulés ils le font passer du côté du biais.

### Questions probables
- *Pourquoi le dropout aide ?* Il empêche les neurones de se reposer les uns sur les autres (co-adaptation) : le
  réseau doit apprendre des motifs plus robustes.
- *L’early stopping, ce n’est pas tricher avec la validation ?* Non : on utilise la validation pour choisir quand
  s’arrêter, comme pour tout réglage. C’est justement pour ça qu’il y a un jeu de test séparé.

---

## A4 · À partir de quelle probabilité dit-on « photo » ? (slide 19)

### L’idée
Le réseau donne une probabilité ; le seuil au-dessus duquel on accepte est un **choix métier**. Plus on est strict,
moins d’intruses passent mais plus on perd de photos. Et sur une archive où les photos sont minoritaires, la
précision baisse mécaniquement.

### Le script
« Le réseau ne dit pas photo ou pas photo : il donne une probabilité, et c’est nous qui fixons le seuil. Par défaut,
0,5. Le graphique montre ce que ça change, sur l’archive réaliste de 4 605 images. En orange, les intruses : des
non-photos acceptées, qui partiraient au captioning. En gris, les vraies photos perdues. À 0,1, on ne perd que 8
photos mais 736 intruses passent. À 0,9, seulement 107 intruses mais on perd 199 photos. Il n’y a pas de bon seuil
dans l’absolu : ça dépend de ce qui coûte le plus à TouNum. À droite, une subtilité : sur le test équilibré, 89 % des
images acceptées sont de vraies photos ; sur l’archive, 76 %. Le réseau n’est pas moins bon, il se trompe même moins
souvent sur les non-photos. Mais elles sont 3,6 fois plus nombreuses que les photos, donc ses erreurs pèsent plus. »

### Les détails
Matrice de confusion à 0,5 (positif = photo) :

| Jeu | Photos gardées (VP) | Intruses (FP) | Photos perdues (FN) | Non-photos écartées (VN) |
|---|---|---|---|---|
| Test équilibré (1 800) | 848 | 108 | 52 | 792 |
| Archive réaliste (4 605) | 953 | 308 | 47 | 3 297 |

| Mesure | Test équilibré | Archive réaliste |
|---|---|---|
| Accuracy | 0,911 | 0,923 |
| Précision = VP / (VP + FP) | 0,887 | 0,756 |
| Rappel = VP / (VP + FN) | 0,942 | 0,953 |
| F1 | 0,914 | 0,843 |
| ROC-AUC | 0,967 | 0,980 |

Effet du seuil sur l’archive réaliste :

| Seuil | 0,1 | 0,3 | 0,5 | 0,7 | 0,9 |
|---|---|---|---|---|---|
| Intruses | 736 | 458 | 308 | 203 | 107 |
| Photos perdues | 8 | 27 | 47 | 89 | 199 |
| Précision | 0,574 | 0,680 | 0,756 | 0,818 | 0,882 |
| Rappel | 0,992 | 0,973 | 0,953 | 0,911 | 0,801 |
| F1 | 0,727 | 0,800 | 0,843 | **0,862** | 0,840 |

Accuracy par type d’image (archive réaliste) : peintures 0,791, schémas 0,901, photos 0,953, dessins 1,000, textes
1,000. Sur les 308 intruses, 209 sont des peintures.

### Les subtilités
- **Le calcul derrière « la précision baisse »** : taux de fausses alarmes sur les non-photos : 108 / 900 = 12 % sur
  le test équilibré, 308 / 3 605 = 8,5 % sur l’archive. Le réseau se trompe donc **moins souvent**. Mais sur
  l’archive, il y a 3 605 non-photos pour 1 000 photos : 8,5 % de 3 605, c’est 308 intruses face à 953 vraies
  photos, soit 24 % des images acceptées.
- **Ce graphique est descriptif** : il est mesuré sur un jeu de test. Un vrai seuil de production devrait être choisi
  sur la validation ou sur de nouvelles données TouNum, puis confirmé. On ne règle jamais un seuil sur le test.
- **Le F1 est le meilleur à 0,7**, pas à 0,5. Mais le F1 suppose que les deux erreurs coûtent pareil, ce que TouNum
  doit décider.
- **L’AUC ne dépend pas du seuil** : elle mesure si le réseau classe les photos au-dessus des non-photos. Elle est
  même plus haute sur l’archive (0,980) : le classement est bon, c’est le seuil et la proportion qui jouent.
- **Ce sont les chiffres de notre CNN** ; le modèle de production, MobileNetV2, fait mieux sur la même archive :
  accuracy 0,971, F1 0,936, peintures 0,938.

### Questions probables
- *Quel seuil recommandez-vous ?* Ça dépend du coût : si légender une peinture est gênant (recherche polluée), monter
  vers 0,7 ; si perdre une photo est grave (client qui paie pour toutes ses photos), rester bas. À fixer avec TouNum
  sur un échantillon de leurs données.
- *Pourquoi MobileNetV2 est meilleur ?* Il a appris à voir sur plus d’un million d’images ImageNet ; on n’entraîne
  que sa dernière couche (1 281 paramètres sur 2,26 millions). Il distingue beaucoup mieux les peintures réalistes.

---

## A5 · Trois débruiteurs : qu’est-ce qui change ? (slide 20)

### L’idée
Trois autoencodeurs de même taille. Les **raccourcis** (U-Net) évitent de faire passer les détails par le goulot ; la
**sortie résiduelle** fait prédire le bruit plutôt que l’image, ce qui laisse intactes les zones propres. Sur les
vraies photos TouNum, seuls ces deux-là font mieux que « ne rien faire ».

### Le script
« Un autoencodeur prend une photo bruitée et rend une photo propre. Les barres montrent la taille de l’image dans le
réseau : elle rétrécit dans l’encodeur, passe par un goulot, puis le décodeur la reconstruit. Version 1, simple :
tout passe par le goulot, au quart de la résolution, donc les détails fins se perdent. Version 2, on ajoute des
raccourcis, comme un U-Net : chaque niveau envoie ses détails directement au décodeur, qui peut se concentrer sur le
bruit. On gagne 4,5 dB. Version 3, celle qu’on garde : le réseau prédit seulement le bruit, et on le soustrait de
l’entrée. Sur une zone déjà propre, il suffit de prédire zéro. Ça compte beaucoup chez TouNum, parce que leur bruit
est très léger. Le chiffre qui compte, c’est « vraies photos » : ne rien faire donne 38,1 dB ; le simple fait pire,
34,3 ; le résiduel est le meilleur, 38,75. »

### Les détails

| Variante | Paramètres | Faux bruit (validation) | Vraies photos, avant réentraînement | Vraies photos, après | Photos propres après* |
|---|---|---|---|---|---|
| Ne rien faire | – | 27,10 dB | 38,07 dB | 38,07 dB | – |
| Filtre gaussien σ = 1 | – | 26,92 dB | 28,05 dB | – | – |
| Autoencodeur simple | 420 515 | 30,03 dB | 32,54 dB | 34,27 dB | 31,7 dB |
| + raccourcis (U-Net) | 466 595 | 34,54 dB | 37,03 dB | 38,49 dB | 45,0 dB |
| **+ sortie résiduelle** | 466 595 | **34,93 dB** | 37,39 dB | **38,75 dB** | **49,0 dB** |

\* ressemblance entre une photo propre et sa sortie : plus c’est haut, moins le réseau l’a modifiée.

Par niveau de bruit synthétique (patchs de validation) :

| σ ajouté | 0 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Photo bruitée | 40,33 | 30,47 | 25,29 | 22,11 | 19,84 |
| Filtre gaussien | 25,84 | 25,62 | 25,06 | 24,29 | 23,43 |
| Autoencodeur résiduel | 38,99 | 36,41 | 34,80 | 33,76 | 32,70 |

- **PSNR** = 10 × log₁₀(1 / erreur quadratique moyenne), pixels entre 0 et 1. +3 dB = erreur divisée par deux.
  Au-dessus de 35 dB, une personne voit à peine le bruit à taille normale.
- **Architecture** : 3 niveaux (32, 64, 128 filtres), à chaque niveau deux convolutions 3×3 + ReLU, MaxPool 2×2 pour
  réduire, Conv2DTranspose 2×2 pour agrandir, sortie Conv 1×1. Perte MSE, Adam 10⁻³, paquets de 32, early stopping
  (patience 6).
- **Entraînement en deux temps** : d’abord sur des photos propres du livrable 1 avec un faux bruit (33 600 patchs
  96 × 96, bruit retiré au hasard à chaque époque), puis réentraînement (fine-tuning) avec un pas dix fois plus petit
  sur un mélange de trois parts égales : recadrages des 78 vraies paires TouNum, de photos propres (la cible est
  l’entrée elle-même), et de faux bruit.
- **Les 148 photos TouNum** : 78 pour le réentraînement, 20 pour la validation (choix du modèle et du seuil), 50 pour
  le test, jamais vues.
- Le modèle retenu améliore 49 photos de test sur 50.

### Les subtilités
- **Le vrai bruit, d’où il vient** : en comparant les fichiers bruités à COCO, on a retrouvé leurs originaux exacts
  (images d’entraînement COCO 1308 à 2349). On a donc la vérité terrain, ce qui est rare. Le bruit est gaussien, de
  moyenne nulle, **uniquement dans les couleurs** (canaux Cb et Cr ; la luminosité Y ne bouge pas plus qu’un simple
  réencodage JPEG), légèrement flou (corrélation 0,45 entre pixels voisins), plus fort sur les couleurs saturées,
  puis enregistré en JPEG qualité 95. Vrai σ médian : 3,2 ; maximum 9,8.
- **Notre faux bruit** reproduit ça : gaussien sur Cb et Cr, flouté par un noyau 3×3, σ tiré entre 0 et 20, puis JPEG
  95. Il ne reproduit pas la perte de saturation ni le bruit plus fort sur les couleurs vives : c’est pour ça qu’on
  réentraîne sur de vraies paires.
- **Pourquoi les réseaux faisaient pire que rien avant réentraînement** : le vrai bruit est très léger (38 dB), donc
  la petite erreur que fait tout réseau, même sur une image propre, est du même ordre que le bruit à retirer.
- **Pourquoi le mélange en trois parts** : réentraîné sur les seules paires TouNum, le résiduel monte à 39,03 dB sur
  les photos bruitées, mais tombe à 35,8 dB sur les photos propres et perd 2 dB sur le faux bruit. Il se
  spécialiserait sur le seul bruit TouNum et abîmerait le reste.
- **Le filtre gaussien fait pire que rien** sur les vraies photos (28,1 dB) : il floute toute l’image, luminosité
  comprise, alors que le bruit TouNum n’est que dans les couleurs.
- **À σ = 0, le débruiteur coûte 1,3 dB** (40,33 → 38,99) : il modifie un peu une image qui n’avait rien à retirer.
  C’est la raison de la règle de l’annexe A6.

### Questions probables
- *Pourquoi les raccourcis n’ajoutent presque pas de paramètres ?* Ils ajoutent des canaux seulement à la première
  convolution de chaque niveau du décodeur : 466 595 contre 420 515.
- *Le réseau marche sur des photos de n’importe quelle taille ?* Oui, il est entièrement convolutif : appris sur des
  patchs 96 × 96, il débruite une photo entière (dimensions complétées à un multiple de 8).
- *Pourquoi pas plus de capacité ?* L’écart entraînement − validation est faible (+0,19 dB) : pas de
  sur-apprentissage, donc plus de capacité ou d’époques aideraient peut-être (côté biais). Le bruit étant redessiné à
  chaque époque, le réseau ne peut pas apprendre ses exemples par cœur.

---

## A6 · Quand faut-il débruiter une photo ? (slide 21)

### L’idée
En production, on n’a pas l’original : on estime le bruit **à l’aveugle** (méthode d’Immerkaer), et on ne débruite
que si l’estimation atteint un seuil. Ce seuil, **3,5**, a été choisi sur les données de validation, pas par une
théorie. Et à ce niveau de bruit, l’estimation reste un signal faible.

### Le script
« En production, on n’a pas l’original propre : il faut décider à l’aveugle si une photo est bruitée. Étape 1, on
estime le bruit avec la méthode d’Immerkaer. Ce petit noyau 3 par 3 annule tout ce qui est lisse dans l’image ; ce
qui reste, c’est surtout le bruit. On fait la moyenne de cette réponse sur tous les pixels, ce qui donne σ, et on
fait la moyenne sur rouge, vert et bleu. Sa limite : les textures fines comme l’herbe passent aussi. Étape 2,
débruiter une photo propre ne peut que l’abîmer un peu : il faut un seuil. Sur 40 images de validation, moitié
bruitées et moitié propres, on a essayé tous les seuils et gardé celui qui fait le plus baisser l’erreur : 3,5.
Étape 3, on vérifie sur le test, honnêtement : la règle fait mieux que ne jamais débruiter, mais débruiter toujours
aurait fait encore mieux. L’estimation sépare mal les photos bruitées et propres à ce niveau de bruit. »

### Les détails
- **Le noyau d’Immerkaer** `[[1, −2, 1], [−2, 4, −2], [1, −2, 1]]` est la différence de deux laplaciens : il donne zéro
  sur une zone constante ou un dégradé régulier.
- **La formule**, pour chaque canal : σ = √(π/2) × (somme des |réponses| / nombre de pixels) / 6. Le 6 normalise le
  noyau ; √(π/2) convertit une moyenne de valeurs absolues en écart-type, pour un bruit gaussien. On fait la moyenne
  des trois σ (R, V, B). Code : `estimate_noise` dans `src/tounum_denoise/data.py`.
- **Le seuil** : sur les 40 images de validation (20 vraies bruitées, 20 photos propres du livrable 1), on calcule
  pour chaque image son σ estimé et son « gain » (baisse de l’erreur quadratique, échelle 0-255, quand on la
  débruite ; négatif pour une photo propre). On teste comme seuils 0 (toujours), chaque σ observé, et l’infini
  (jamais), et on garde celui qui maximise le gain moyen : 3,4879…, arrondi à 3,5. Il est sauvegardé dans
  `models/denoiser.json` (`noise_threshold`).
- **Résultat en test** (50 bruitées + 50 propres) : gain moyen 0 en ne débruitant jamais, +0,21 avec la règle, +0,47
  en débruitant toujours. La règle débruite 58 % des bruitées et 64 % des propres. Débruiter une photo propre la
  dégrade toujours un peu (−0,82 en moyenne), une bruitée gagne +1,76.

### Les subtilités
- **3,5 n’est ni une somme ni une moyenne de Y, Cb, Cr** : c’est un σ en niveaux 0-255, moyenné sur R, V, B.
  L’estimateur ne sait pas que le bruit a été mis dans Cb et Cr, il regarde le fichier tel quel.
- **L’estimation est faible** : sur les 148 photos, sa corrélation avec le vrai bruit n’est que de 0,38. Sur les
  originaux **propres**, elle donne une médiane de 4,5, contre 4,6 sur les fichiers bruités : la texture de l’image
  pèse plus que le bruit lui-même.
- **Pourquoi garder la règle si « toujours » gagne en test ?** On n’a pas le droit de choisir sur le test (sinon le
  test ne mesure plus rien). Et l’écart coûte peu : sur une photo propre, le débruiteur retenu ne fait presque rien
  (49 dB de ressemblance), et pour les légendes ça ne change rien (BLEU identique, annexe A8).
- **Le mélange 50/50 est une hypothèse** : TouNum ne nous a pas dit quelle part de leurs photos est bruitée. Avec
  beaucoup plus de photos propres, le seuil optimal monterait ; avec beaucoup plus de bruitées, il descendrait.
- **Le filtre photo du module 1 supporte le bruit** : accuracy 0,967 sur les images d’origine, 0,943 à σ = 20, 0,960
  si on débruite d’abord. La réduction en 128 × 128 moyenne l’essentiel du bruit. C’est pour ça qu’on peut classer
  **avant** de débruiter, et ne débruiter que les photos.

### Questions probables
- *Pourquoi ne pas tout débruiter ?* Un débruiteur ne peut pas ajouter d’information : sur une photo propre, il ne
  peut que modifier des détails. Le seuil évite de toucher inutilement aux photos qui vont bien.
- *Que feriez-vous avec plus de temps ?* Demander à TouNum quels lots viennent d’un scanner bruité : c’est le signal
  le plus fiable. Ou un estimateur de bruit appris, plutôt qu’un noyau fixe.

---

## A7 · Le modèle de légende, pièce par pièce (slide 22)

### L’idée
Un **œil** déjà entraîné (InceptionV3, qu’on ne touche pas) décrit la photo en 64 zones ; une **plume** (un réseau
récurrent GRU) écrit la légende mot par mot. On entraîne la plume à deviner le mot suivant en lui donnant les vrais
mots précédents ; à l’usage, elle repart de ses propres mots.

### Le script
« On suit une photo de gauche à droite. Elle est mise en 299 par 299 pixels, la taille qu’attend InceptionV3. C’est
l’œil : un réseau déjà entraîné sur 1,2 million d’images. On le garde gelé, on ne touche à aucun de ses 21,8
millions de poids. Il découpe la photo en une grille de 8 par 8, soit 64 zones, et décrit chacune par 2 048 nombres.
Une couche qu’on entraîne ramène chaque zone à 256 nombres, et leur moyenne donne à la plume son point de départ. La
plume est un GRU : à chaque pas, il lit le mot précédent et donne un score pour chacun des 5 000 mots du
vocabulaire ; le plus haut est le mot suivant. À l’entraînement, on lui donne toujours les vrais mots précédents,
c’est le teacher forcing. À l’usage, il repart de ses propres mots, en prenant à chaque fois le plus probable : c’est
le décodage glouton. Une erreur au début peut donc entraîner les suivantes. »

### Les détails
- **Encodeur** : InceptionV3 sans sa tête de classification, gelé ; sortie du dernier bloc convolutif 8 × 8 × 2 048,
  aplatie en 64 vecteurs de 2 048 valeurs. Pixels ramenés dans [−1, 1] dans le modèle.
- **Projection** : Dense 256 + ReLU sur chaque zone. État initial du GRU : moyenne des zones → LayerNorm → Dense 512
  + tanh.
- **Décodeur** : Embedding 256 (un vecteur par mot) → GRU 512 → Dense 512 + ReLU + dropout 0,3 → Dense 5 000 (un score
  par mot), softmax.
- **Paramètres** : 5 947 016 entraînés pour le modèle retenu (sans attention) ; 6 209 672 avec attention, dont
  131 584 pour l’attention.
- **Texte** : minuscules, lettres seulement, marqueurs `startseq` et `endseq` ; 5 000 mots les plus fréquents (98,7 %
  des occurrences ; 22 431 mots distincts au départ), les autres deviennent `[UNK]`. Légendes coupées à 22 tokens
  (99,4 % y tiennent ; médiane 10 mots).
- **Entraînement** : perte = entropie croisée sur le mot suivant (−log p du bon mot), padding ignoré ; Adam 10⁻³,
  gradient plafonné à 5 ; paquets de 64 ; une légende tirée au hasard par image et par époque ; early stopping
  (patience 6). 80 000 images COCO d’entraînement, 2 000 de validation, 2 000 de test (COCO val2017).
- **Génération** : on part de `startseq`, on garde le mot le plus probable, jusqu’à `endseq` ou 20 mots ; `[UNK]` et
  le padding ne sont jamais choisis.

### Les subtilités
- **Pourquoi geler InceptionV3** : ses features ne changent jamais, donc on les calcule une seule fois et on les met
  en cache (22 Go en flottants 16 bits pour 84 000 images). Chaque époque ne lit que ce cache : 18 secondes par
  époque. C’est aussi la réponse à la contrainte mémoire de l’énoncé : le décodeur ne charge jamais les images.
- **Teacher forcing vs usage** : à l’entraînement, le réseau ne voit jamais ses propres erreurs ; à l’usage, il ne
  voit que les siennes. C’est la principale différence entre les deux (on parle de *biais d’exposition*).
- **La LayerNorm** avant l’état initial n’est pas décorative : sans elle, la tanh saturait et une première version
  donnait le même état de départ à toutes les images ; le modèle sans attention ignorait alors l’image.
- **ImageNet ne connaît que des objets** : InceptionV3 sait nommer un objet, pas décrire une action ou une scène.
  C’est le rôle de la plume, apprise sur les légendes COCO.
- **Les 148 photos TouNum** sont des images d’entraînement COCO : on a retiré leurs originaux (102 tombaient dans
  l’échantillon) pour que l’étude de cas et la démo portent sur des images jamais vues.

### Questions probables
- *GRU ou LSTM ?* Même rôle : des portes décident quoi garder des mots précédents. Le GRU a moins de paramètres.
- *Pourquoi plafonner le gradient ?* Les réseaux récurrents ont parfois des gradients très grands qui font diverger
  l’apprentissage ; on limite leur norme à 5.
- *Pourquoi en anglais ?* COCO est annoté en anglais. Pour TouNum, il faudrait ajouter une traduction automatique ou
  un corpus de légendes en français.

---

## A8 · Comment note-t-on une légende ? (slide 23)

### L’idée
Le score **BLEU** compte les groupes de mots communs avec cinq légendes humaines. Il sert à **comparer des modèles
entre eux**, pas à dire si une légende est juste. L’attention ne change rien au score, donc on garde le modèle le
plus léger. Les légendes restent génériques.

### Le script
« Comment noter une légende automatiquement ? Le score standard, c’est BLEU. On compte combien de groupes de 1 à 4
mots qui se suivent se retrouvent dans l’une des cinq légendes écrites par des humains pour la même photo. Le score
va de 0 à 1. BLEU-4 exige des suites de 4 mots, donc il juge aussi l’ordre et la formulation. Notre modèle fait 0,252
sur 2 000 photos de test. Pour avoir une échelle, on a noté une légende humaine de la même façon, face aux quatre
autres : 0,186. Ça ne veut pas dire que le modèle écrit mieux qu’un humain : l’humain varie ses mots, alors que BLEU
récompense les tournures courantes. Avec ou sans attention : 0,254 contre 0,252, égalité, donc on garde le plus
léger. Plus d’images aide encore, mais de moins en moins. Et les phrases restent génériques : 503 mots différents
contre 2 279 chez les humains, et 30 % des légendes reprennent mot pour mot une légende d’entraînement. »

### Les détails

| Test (2 000 images) | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| Avec attention | 0,683 | 0,503 | 0,357 | 0,254 |
| Sans attention (retenu) | 0,687 | 0,504 | 0,357 | 0,252 |
| Une légende humaine face aux 4 autres | 0,630 | 0,428 | 0,283 | 0,186 |

| Images d’entraînement | 10 000 | 20 000 | 40 000 | 80 000 |
|---|---|---|---|---|
| BLEU-4 (validation) | 0,215 | 0,243 | 0,253 | 0,256 |
| Perte de validation | 2,801 | 2,614 | 2,479 | 2,377 |

- **BLEU de corpus** : calculé sur tout le jeu de test d’un coup, avec une pénalité pour les phrases trop courtes.
- **Choix du modèle** sur les 2 000 images de validation : BLEU-4 de 0,256 pour les deux versions ; la version sans
  attention a été retenue (5,95 M paramètres contre 6,21 M).
- Longueur moyenne : 9,7 mots (humains : 10,5).
- **Robustesse au bruit** (1 000 images de test, faux bruit fort) : BLEU-4 de 0,257 sur les images propres ; à σ = 20,
  0,249 sur l’image bruitée et 0,257 si on débruite d’abord. Moins de 0,01 d’écart.
- **Workflow complet** sur le dossier de démo (18 fichiers) : 17 décisions sur 18 justes, 6 photos débruitées,
  0,36 s par image.

### Les subtilités
- **Pourquoi l’humain a un score plus bas** : il n’a que 4 références au lieu de 5 (un peu désavantagé), et surtout
  chaque personne décrit la scène avec ses propres mots. BLEU récompense le recouvrement de mots, donc les formulations
  les plus fréquentes, exactement ce que produit le décodage glouton.
- **Pourquoi la slide 12 montre l’attention alors qu’on ne la garde pas ?** Elle illustre le principe et rend visibles
  les choix du modèle. Sur le score, les deux versions sont à égalité. Dire franchement que le modèle en production
  est celui sans attention.
- **Pourquoi l’attention n’aide pas ici** : chacune des 64 zones d’InceptionV3 vient du dernier bloc, qui « voit »
  déjà une grande partie de l’image. Le modèle n’a pas besoin de regarder l’objet pour le nommer ; c’est aussi pour
  ça que les cartes d’attention réelles sont diffuses.
- **Pendant l’entraînement**, le BLEU-4 sur 500 images de validation donnait 0,256 avec attention contre 0,239 sans.
  Sur les 2 000 images de validation complètes, c’est l’égalité. Leçon : 500 légendes, c’est trop peu pour départager.
- **La courbe s’aplatit sur le BLEU** (+0,003 de 40 000 à 80 000 images), mais la perte de validation baisse encore
  de 0,10 : plus de données aiderait encore (côté variance). Le BLEU est une mesure plus bruitée.
- **Sur les 50 photos TouNum**, le BLEU-4 est de 0,328 sur les originaux et 0,340 sur les fichiers bruités : le bruit
  ne fait pas mieux, c’est la variabilité d’une mesure sur 50 images seulement. 14 % des légendes changent entre
  l’original et le fichier bruité.

### Questions probables
- *Qu’est-ce que le beam search ?* Au lieu de garder un seul mot à chaque pas, on garde les k meilleures phrases
  partielles et on choisit la meilleure à la fin. Les légendes seraient moins génériques.
- *D’autres métriques ?* METEOR, CIDEr ou SPICE tiennent mieux compte des synonymes et du sens ; BLEU reste le
  standard de comparaison. Le vrai juge pour TouNum serait un petit jeu de leurs photos légendées à la main.
- *Comment améliorer le modèle ?* Beam search, plus d’images (tout COCO), fine-tuning des derniers blocs
  d’InceptionV3 ou un encodeur plus récent.
