SCREEN_TO_DISEASE = {
    "autism": "Autisme",
    "alzheimer": "Alzheimer",
    "brain_stroke": "AVC",
    "ms": "SEP",
    "parkinson": "Parkinson",
    "brain_tumor": "Tumeur cérébrale"
}

DISEASE_INFO = {
    "Autisme": {
        "definition": "L'autisme est un trouble du développement neurologique qui apparaît généralement dès la petite enfance. Il se manifeste par des difficultés dans la communication, des comportements répétitifs et une interaction sociale limitée. Les causes exactes de l'autisme sont encore mal comprises, mais des facteurs génétiques et environnementaux sont impliqués.",
        "symptoms": "Défis sociaux, comportements répétitifs, difficultés de communication, intérêts restreints."
    },
    "Alzheimer": {
        "definition": "La maladie d'Alzheimer est une affection neurodégénérative progressive qui détruit lentement la mémoire et les compétences cognitives. Elle perturbe la capacité à effectuer les tâches quotidiennes. Cette maladie est la forme la plus courante de démence chez les personnes âgées.",
        "symptoms": "Perte de mémoire, confusion, désorientation, troubles du langage, difficulté à reconnaître les proches."
    },
    "AVC": {
        "definition": "Un accident vasculaire cérébral (AVC) se produit lorsque l'irrigation sanguine du cerveau est interrompue, entraînant la mort des cellules cérébrales. Il peut être ischémique (obstruction d'une artère) ou hémorragique (rupture d'un vaisseau sanguin). L'AVC est une urgence médicale majeure.",
        "symptoms": "Engourdissement soudain, perte de la parole, vision trouble, paralysie d'un côté du corps, maux de tête intenses."
    },
    "SEP": {
        "definition": "La sclérose en plaques (SEP) est une maladie auto-immune chronique qui affecte le système nerveux central. Elle est caractérisée par une attaque du système immunitaire contre la myéline, une substance qui protège les fibres nerveuses, entraînant des troubles neurologiques progressifs.",
        "symptoms": "Fatigue, troubles de la marche, vision double, engourdissements, troubles de l'équilibre."
    },
    "Parkinson": {
        "definition": "La maladie de Parkinson est une pathologie neurodégénérative affectant principalement les mouvements. Elle résulte de la perte progressive des cellules dopaminergiques dans le cerveau. Cette maladie évolue lentement et entraîne des troubles moteurs et non moteurs.",
        "symptoms": "Tremblements, rigidité musculaire, lenteur des mouvements, perte d'équilibre, troubles du sommeil."
    },
    "Tumeur cérébrale": {
        "definition": "Une tumeur cérébrale est une masse de cellules anormales dans le cerveau. Elle peut être bénigne ou maligne, et selon sa localisation, elle peut affecter des fonctions vitales comme la parole, la vision, ou le contrôle moteur. Le traitement dépend du type et de la taille de la tumeur.",
        "symptoms": "Céphalées, convulsions, changements de personnalité, troubles de la vision, nausées."
    }
}

MODEL_EXPLANATIONS = {
    "CNN": (
        "CNN (Convolutional Neural Network) : "
        "Un réseau de neurones artificiels spécialisé dans le traitement des images. "
        "Les CNN utilisent des couches de convolution pour extraire automatiquement des caractéristiques visuelles pertinentes, "
        "comme les formes, les textures et les motifs, à partir des images médicales. "
        "Ils sont particulièrement efficaces pour la classification et la détection d'anomalies dans les images cérébrales."
    ),
    "ViT": (
        "ViT (Vision Transformer) : "
        "Un modèle de deep learning basé sur l'architecture Transformer, initialement conçue pour le traitement du langage naturel. "
        "Le ViT divise une image en petits patchs, les traite comme une séquence, et utilise des mécanismes d'attention pour capturer "
        "les relations globales entre différentes régions de l'image. Il est reconnu pour ses performances élevées en classification d'images complexes."
    ),
    "U-Net": (
        "U-Net : "
        "Une architecture de réseau de neurones conçue spécifiquement pour la segmentation d'images médicales. "
        "Le U-Net possède une structure en forme de U, avec un chemin d'encodage pour capturer le contexte et un chemin de décodage pour une localisation précise. "
        "Il permet d'identifier et de délimiter avec précision les régions d'intérêt, comme les lésions ou tumeurs cérébrales."
    ),
    "GCN": (
        "GCN (Graph Convolutional Network) : "
        "Un réseau de neurones adapté à l'analyse de données structurées en graphes, où les relations entre les éléments sont aussi importantes que les éléments eux-mêmes. "
        "Les GCN sont utilisés pour modéliser des connexions complexes, par exemple entre différentes régions du cerveau, et sont efficaces pour détecter des schémas anormaux dans les réseaux neuronaux."
    ),
    "VGG19": (
        "VGG19 : "
        "Un réseau de neurones profond classique composé de 19 couches, connu pour sa simplicité et sa robustesse. "
        "VGG19 utilise de nombreuses couches de convolution avec de petits filtres pour extraire progressivement des caractéristiques de plus en plus complexes. "
        "Il est largement utilisé pour la classification d'images médicales et la reconnaissance de motifs pathologiques."
    )
} 