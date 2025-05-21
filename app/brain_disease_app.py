from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window

Window.clearcolor = (0.95, 0.95, 0.97, 1)

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

# Load the style.kv file
Builder.load_file('style.kv')

class MainMenu(Screen):
    pass

class DiseaseBaseScreen(Screen):
    def open_filechooser(self):
        from kivy.uix.filechooser import FileChooserListView
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button

        content = BoxLayout(orientation='vertical')
        filechooser = FileChooserListView(filters=['*.png', '*.jpg', '*.jpeg'])
        content.add_widget(filechooser)

        btn = Button(text='Sélectionner', size_hint_y=0.1)
        btn.bind(on_press=lambda x: self.select_image(filechooser.path, filechooser.selection))
        content.add_widget(btn)

        self.popup = Popup(title="Sélectionner une image cérébrale", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def select_image(self, path, selection):
        if selection:
            self.ids.uploaded_image.source = selection[0]
        self.popup.dismiss()

    def analyze_image(self):
        self.ids.result_label.text = "Analyse terminée ! (résultat simulé)"

class AutismScreen(DiseaseBaseScreen):
    pass

class AlzheimerScreen(DiseaseBaseScreen):
    pass

class BrainStrokeScreen(DiseaseBaseScreen):
    pass

class MScreen(DiseaseBaseScreen):
    pass

class ParkinsonScreen(DiseaseBaseScreen):
    pass

class BrainTumorScreen(DiseaseBaseScreen):
    pass

class DiseaseScreen(Screen):
    def __init__(self, disease_name, **kwargs):
        super().__init__(**kwargs)
        self.disease_name = disease_name

    def on_enter(self):
        self.ids.disease_title.text = self.disease_name
        self.update_disease_info()

    def update_disease_info(self):
        info = DISEASE_INFO.get(self.disease_name, {})
        text = f"[b]Définition :[/b]\n{info.get('definition', '')}\n\n[b]Symptômes :[/b]\n{info.get('symptoms', '')}"
        self.ids.disease_info.text = text

class BrainDiseaseApp(App):
    def build(self):
        self.title = 'Analyseur Cérébral'
        sm = ScreenManager()
        sm.add_widget(MainMenu(name='menu'))
        sm.add_widget(AutismScreen(name='autism'))
        sm.add_widget(AlzheimerScreen(name='alzheimer'))
        sm.add_widget(BrainStrokeScreen(name='brain_stroke'))
        sm.add_widget(MScreen(name='ms'))
        sm.add_widget(ParkinsonScreen(name='parkinson'))
        sm.add_widget(BrainTumorScreen(name='brain_tumor'))

        for disease in DISEASE_INFO.keys():
            screen_name = disease.lower().replace(' ', '_')
            sm.add_widget(DiseaseScreen(disease, name=screen_name))

        sm.current = 'menu'
        return sm

if __name__ == '__main__':
    BrainDiseaseApp().run()