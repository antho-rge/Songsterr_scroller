# Guitar AutoScroll

Un script Python léger permettant de faire défiler automatiquement des partitions (comme Songsterr) lors de la pratique de la guitare, en utilisant le suivi du regard (Eye Tracking).

## Prérequis

- Une webcam
- Python 3.8 ou supérieur

## Installation

1. Cloner le dépôt :
\`\`\`bash
git clone https://github.com/TON_PSEUDO/guitar-autoscroll.git
cd guitar-autoscroll
\`\`\`

2. Créer et activer un environnement virtuel :
\`\`\`bash
python -m venv venv
# Windows : venv\Scripts\activate
# Mac/Linux : source venv/bin/activate
\`\`\`

3. Installer les dépendances :
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Utilisation

Lancer le script principal :
\`\`\`bash
python main.py
\`\`\`

**Processus de calibration :**
Au lancement, le script nécessite une calibration rapide en deux étapes (3 secondes chacune) :
1. Regarder normalement l'écran.
2. Regarder le bas de l'écran (vers la dernière ligne de la partition).

Une fois calibré, le script appuiera automatiquement sur "Page Down" lorsque le regard fixera le bas de l'écran.
Appuyez sur `r` pour recalibrer, ou `q` pour quitter.