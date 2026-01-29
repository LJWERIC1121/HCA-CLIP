import os
import json
from collections import defaultdict

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader


# Simple template for baseline comparison
simple_template = "a photo of {}"

# Detailed prompts for each plant disease class (10 prompts per class)
# Keys use the format with spaces (as classnames are converted from underscores to spaces)
detailed_prompts = {
    'Apple   Apple scab': [
        'A leaf showing dark, irregular spots characteristic of apple scab disease',
        'Apple fruit with rough, circular lesions caused by scab fungus',
        'An apple tree leaf displaying olive-green to brown velvety patches of scab infection',
        'Close-up of apple foliage with raised, dark brown scab lesions',
        'An infected apple with corky, scabbed surface areas',
        'Apple leaves showing premature yellowing and spots from scab disease',
        'Young apple fruit with small black spots indicating early scab infection',
        'Mature apple with extensive scabbing and deformed growth from disease',
        'Apple tree branch with multiple leaves showing scab disease symptoms',
        'Overhead view of apple leaves with scattered dark scab lesions'
    ],
    'Apple   Black rot': [
        'An apple with large, dark brown rotted areas and concentric rings',
        'Apple leaves displaying reddish-brown spots with purple margins',
        'Severely infected apple showing black rot with mummified appearance',
        'Apple fruit with characteristic bull\'s eye pattern from black rot',
        'Early stage black rot on apple showing small purple spots',
        'Apple branch with leaves exhibiting frogeye leaf spot symptoms',
        'Rotted apple with black pycnidia forming on the surface',
        'Apple tree foliage with numerous circular lesions from black rot',
        'Cross-section of apple showing internal brown rot decay',
        'Multiple apples on tree displaying various stages of black rot infection'
    ],
    'Apple   Cedar apple rust': [
        'Apple leaves with bright orange-yellow spots from cedar rust',
        'Underside of apple leaf showing spore-producing structures of rust fungus',
        'Young apple fruit with small rust lesions and distorted growth',
        'Apple tree leaves covered in yellow-orange rust pustules',
        'Close-up of rust fungus causing raised orange spots on leaf surface',
        'Apple foliage in spring showing early cedar rust infection',
        'Severely infected apple leaves with multiple rust lesions causing defoliation',
        'Apple with small, sunken rust spots on the fruit surface',
        'Cluster of apple leaves displaying characteristic rust disease coloration',
        'Apple tree branch with leaves showing orange rust galls'
    ],
    'Apple   healthy': [
        'A healthy apple tree with vibrant green leaves and no disease symptoms',
        'Close-up of unblemished, smooth green apple leaves',
        'Fresh, healthy apples with glossy red skin on the tree',
        'Apple orchard with healthy trees showing uniform green foliage',
        'Perfect apple leaves with consistent green color and no spots',
        'Healthy apple branch with clean, disease-free fruit development',
        'Lush green apple tree canopy indicating optimal health',
        'Close-up of healthy apple showing smooth, unblemished skin',
        'Young healthy apple leaves with proper shape and color',
        'Well-maintained apple tree displaying robust growth and healthy appearance'
    ],
    'Blueberry   healthy': [
        'Healthy blueberry bush with rich green foliage',
        'Blueberry plant displaying uniform green leaves without blemishes',
        'Close-up of healthy blueberry leaves showing proper color and texture',
        'Blueberry branch with healthy leaves and developing fruit',
        'Pristine blueberry foliage with no signs of disease or damage',
        'Mature blueberry plant with dense, healthy green leaves',
        'Young blueberry shoots showing vigorous healthy growth',
        'Blueberry bush with glossy green leaves in optimal condition',
        'Healthy blueberry plantation with uniform plant growth',
        'Close-up of unblemished blueberry leaves with proper venation'
    ],
    'Cherry (including sour)   Powdery mildew': [
        'Cherry leaves covered with white powdery fungal growth',
        'Cherry tree foliage showing characteristic powdery mildew coating',
        'Young cherry leaves curled and distorted from mildew infection',
        'Cherry branch with leaves displaying white fungal patches',
        'Close-up of powdery mildew spores forming white dust on leaf surface',
        'Severely infected cherry leaves with dense white mildew coverage',
        'Cherry tree showing widespread powdery mildew on new growth',
        'Cherry leaves with mildew causing leaf curl and stunted development',
        'Cherry foliage with patches of white powdery fungus',
        'Late-stage powdery mildew infection on cherry causing leaf browning'
    ],
    'Cherry (including sour)   healthy': [
        'Healthy cherry tree with deep green leaves and no disease',
        'Close-up of pristine cherry leaves showing proper color and form',
        'Cherry branch with healthy foliage and developing fruit',
        'Vibrant green cherry tree canopy indicating good health',
        'Unblemished cherry leaves with smooth edges and uniform color',
        'Healthy cherry orchard with trees showing vigorous growth',
        'Young cherry leaves in perfect condition without spots or damage',
        'Mature cherry tree displaying robust healthy foliage',
        'Close-up of healthy cherry branch with clean leaves',
        'Cherry tree with lush green foliage and optimal health status'
    ],
    'Corn (maize)   Cercospora leaf spot Gray leaf spot': [
        'Corn leaves with rectangular gray lesions parallel to leaf veins',
        'Corn plant showing multiple gray leaf spot symptoms',
        'Close-up of corn leaf with characteristic gray-brown rectangular spots',
        'Severely infected corn with extensive gray leaf spot damage',
        'Corn foliage displaying early stage gray leaf spot lesions',
        'Corn field with plants affected by gray leaf spot disease',
        'Mature corn leaves with large gray necrotic areas',
        'Corn plant showing premature yellowing from gray leaf spot',
        'Close-up of corn leaf veins with gray spot lesions between them',
        'Corn crop with widespread gray leaf spot causing reduced photosynthesis'
    ],
    'Corn (maize)   Common rust ': [
        'Corn leaves with reddish-brown pustules erupting through leaf surface',
        'Corn plant heavily infected with common rust showing numerous pustules',
        'Close-up of corn leaf with circular to elongated rust pustules',
        'Corn foliage displaying characteristic orange-brown rust spores',
        'Severely rusted corn leaves with pustules on both sides',
        'Young corn plant showing early rust infection symptoms',
        'Corn field with plants affected by common rust disease',
        'Mature corn leaves covered with rust pustules releasing spores',
        'Corn stalk and leaves displaying extensive rust infection',
        'Close-up of corn rust pustules breaking through leaf epidermis'
    ],
    'Corn (maize)   Northern Leaf Blight': [
        'Corn leaves with large, elliptical gray-green lesions',
        'Corn plant showing characteristic cigar-shaped northern blight lesions',
        'Severely infected corn with extensive leaf blight damage',
        'Close-up of corn leaf with long tan to gray lesions',
        'Corn foliage displaying northern leaf blight with distinct margins',
        'Young corn plant with early blight lesions developing',
        'Corn field affected by northern leaf blight showing reduced yield',
        'Mature corn leaves with large necrotic areas from blight',
        'Corn stalk with leaves displaying progressive blight symptoms',
        'Close-up of northern leaf blight lesions spanning across corn leaf'
    ],
    'Corn (maize)   healthy': [
        'Healthy corn plant with vibrant green leaves',
        'Close-up of unblemished corn leaves showing proper color',
        'Corn field with healthy plants displaying uniform growth',
        'Young corn plant with perfect green foliage',
        'Mature corn with healthy leaves and developing ears',
        'Corn crop showing optimal health and vigor',
        'Close-up of healthy corn leaf with no spots or lesions',
        'Corn plant with lush green foliage indicating good nutrition',
        'Healthy corn field with plants at optimal growth stage',
        'Corn leaves with consistent green color and no disease symptoms'
    ],
    'Grape   Black rot': [
        'Grape leaves with circular brown spots and dark margins',
        'Grape berries shriveled and mummified from black rot',
        'Grape cluster with some berries showing black rot symptoms',
        'Close-up of grape leaf with characteristic black rot lesions',
        'Infected grape showing black pycnidia on fruit surface',
        'Grape vine with leaves displaying multiple black rot spots',
        'Severely affected grape bunch with dried, blackened berries',
        'Early stage black rot on grape leaves showing tan spots',
        'Grape foliage with extensive black rot damage',
        'Mummified grape berries hanging on vine from black rot'
    ],
    'Grape   Esca (Black Measles)': [
        'Grape leaves with tiger stripe pattern from esca disease',
        'Grape vine showing characteristic black measles symptoms on fruit',
        'Grape berries with dark spots and shriveling from esca',
        'Close-up of grape leaf with interveinal necrosis from esca',
        'Severely infected grape showing black spotting on berries',
        'Grape foliage displaying yellowing and necrosis from esca',
        'Grape cluster with berries showing measles-like dark spots',
        'Grape vine with leaves exhibiting characteristic esca striping',
        'Mature grape leaves with extensive esca disease damage',
        'Grape bunch with some berries affected by black measles'
    ],
    'Grape   Leaf blight (Isariopsis Leaf Spot)': [
        'Grape leaves with angular brown spots from leaf blight',
        'Grape vine foliage showing isariopsis leaf spot symptoms',
        'Close-up of grape leaf with irregular brown lesions',
        'Severely infected grape leaves with extensive blight damage',
        'Grape plant displaying multiple leaf spot lesions',
        'Young grape leaves with early blight infection',
        'Grape foliage with spots causing premature defoliation',
        'Mature grape leaves with large necrotic areas from blight',
        'Grape vine with widespread leaf spot disease',
        'Close-up of grape leaf showing characteristic blight lesions'
    ],
    'Grape   healthy': [
        'Healthy grape vine with lush green foliage',
        'Close-up of pristine grape leaves without spots or damage',
        'Grape cluster with healthy berries developing normally',
        'Vineyard with healthy grape plants showing uniform growth',
        'Unblemished grape leaves with proper color and texture',
        'Healthy grape vine with robust foliage and fruit',
        'Young grape leaves in perfect condition',
        'Mature grape plant displaying optimal health',
        'Close-up of healthy grape bunch with firm berries',
        'Grape vineyard with plants showing vigorous healthy growth'
    ],
    'Orange   Haunglongbing (Citrus greening)': [
        'Orange tree with yellowing leaves from citrus greening disease',
        'Orange fruit showing lopsided shape from huanglongbing',
        'Close-up of orange leaves with characteristic blotchy mottling',
        'Severely infected orange tree with thinning canopy from greening',
        'Orange with green, bitter fruit from citrus greening',
        'Young orange leaves showing asymmetric yellowing pattern',
        'Orange branch with leaves displaying HLB symptoms',
        'Mature orange tree declining from huanglongbing infection',
        'Orange fruit remaining small and green due to greening disease',
        'Orange tree foliage with irregular yellow patterns from HLB'
    ],
    'Peach   Bacterial spot': [
        'Peach leaves with small dark spots surrounded by yellow halos',
        'Peach fruit showing raised lesions from bacterial spot',
        'Close-up of peach leaf with numerous bacterial spot lesions',
        'Severely infected peach with extensive leaf spotting',
        'Young peach leaves with early bacterial spot symptoms',
        'Peach branch with leaves showing shot-hole appearance from spots',
        'Peach fruit with dark, sunken bacterial spot lesions',
        'Mature peach leaves with extensive bacterial damage',
        'Peach tree foliage with widespread bacterial spot infection',
        'Close-up of peach leaf with holes from dropped spot lesions'
    ],
    'Peach   healthy': [
        'Healthy peach tree with vibrant green foliage',
        'Close-up of unblemished peach leaves showing proper color',
        'Peach branch with healthy leaves and developing fruit',
        'Peach orchard with healthy trees displaying uniform growth',
        'Pristine peach leaves without spots or damage',
        'Young healthy peach tree with perfect foliage',
        'Mature peach plant showing robust health',
        'Close-up of healthy peach fruit on tree',
        'Peach tree with lush green canopy',
        'Healthy peach leaves with consistent color and no lesions'
    ],
    'Pepper, bell   Bacterial spot': [
        'Pepper leaves with small dark spots from bacterial infection',
        'Bell pepper fruit showing raised lesions from bacterial spot',
        'Close-up of pepper leaf with numerous bacterial spot symptoms',
        'Severely infected pepper plant with extensive leaf spotting',
        'Young pepper leaves with early bacterial spot lesions',
        'Pepper fruit with dark, scabby spots from bacterial disease',
        'Pepper plant foliage with widespread bacterial spot damage',
        'Mature pepper leaves with extensive bacterial lesions',
        'Bell pepper showing deformed growth from bacterial spot',
        'Close-up of pepper leaf with characteristic bacterial spotting'
    ],
    'Pepper, bell   healthy': [
        'Healthy bell pepper plant with vibrant green leaves',
        'Close-up of unblemished pepper leaves showing proper color',
        'Pepper plant with healthy foliage and developing fruit',
        'Bell pepper crop with healthy plants displaying uniform growth',
        'Pristine pepper leaves without spots or damage',
        'Young healthy pepper plant with perfect foliage',
        'Mature pepper displaying robust health and fruit production',
        'Close-up of healthy bell pepper fruit on plant',
        'Pepper plant with lush green foliage',
        'Healthy pepper leaves with consistent color and no lesions'
    ],
    'Potato   Early blight': [
        'Potato leaves with circular brown spots showing concentric rings',
        'Potato plant displaying characteristic target spot lesions',
        'Close-up of potato leaf with early blight bull\'s eye pattern',
        'Severely infected potato with extensive early blight damage',
        'Potato foliage showing premature yellowing from early blight',
        'Young potato plant with initial early blight symptoms',
        'Potato field with plants affected by early blight disease',
        'Mature potato leaves with large necrotic areas from blight',
        'Potato plant showing progressive early blight infection',
        'Close-up of potato leaf with characteristic concentric ring lesions'
    ],
    'Potato   Late blight': [
        'Potato leaves with dark, water-soaked lesions from late blight',
        'Potato plant showing rapid tissue death from late blight',
        'Close-up of potato leaf with characteristic late blight lesions',
        'Severely infected potato field with widespread late blight',
        'Potato foliage with white fuzzy growth on lesion undersides',
        'Young potato plant collapsing from late blight infection',
        'Potato stems showing dark streaks from late blight',
        'Mature potato leaves with extensive late blight damage',
        'Potato plant in advanced stage of late blight decay',
        'Close-up of potato leaf with water-soaked late blight spots'
    ],
    'Potato   healthy': [
        'Healthy potato plant with lush green foliage',
        'Close-up of unblemished potato leaves showing proper color',
        'Potato field with healthy plants displaying uniform growth',
        'Pristine potato leaves without spots or lesions',
        'Young healthy potato plant with vigorous growth',
        'Mature potato displaying robust health and tuber development',
        'Potato crop showing optimal health and yield potential',
        'Close-up of healthy potato leaves with proper structure',
        'Potato plant with dense green foliage',
        'Healthy potato field with plants at peak growth'
    ],
    'Raspberry   healthy': [
        'Healthy raspberry plant with vibrant green leaves',
        'Close-up of unblemished raspberry foliage',
        'Raspberry canes with healthy leaves and fruit development',
        'Pristine raspberry leaves without disease symptoms',
        'Young healthy raspberry plant showing vigorous growth',
        'Mature raspberry displaying robust health',
        'Raspberry patch with healthy plants and uniform growth',
        'Close-up of healthy raspberry leaves with proper color',
        'Raspberry plant with lush green foliage and berries',
        'Healthy raspberry canes with optimal leaf condition'
    ],
    'Soybean   healthy': [
        'Healthy soybean plant with vibrant green trifoliate leaves',
        'Close-up of unblemished soybean foliage',
        'Soybean field with healthy plants displaying uniform growth',
        'Pristine soybean leaves without spots or damage',
        'Young healthy soybean plant with perfect foliage',
        'Mature soybean displaying robust health and pod development',
        'Soybean crop showing optimal health and yield potential',
        'Close-up of healthy soybean leaves with proper structure',
        'Soybean plant with lush green foliage',
        'Healthy soybean field at flowering stage'
    ],
    'Squash   Powdery mildew': [
        'Squash leaves covered with white powdery fungal coating',
        'Squash plant showing extensive powdery mildew infection',
        'Close-up of squash leaf with characteristic white mildew patches',
        'Severely infected squash with leaves fully covered in mildew',
        'Young squash leaves showing early powdery mildew symptoms',
        'Squash vine with leaves displaying white fungal growth',
        'Mature squash leaves with dense powdery mildew coverage',
        'Squash plant with mildew causing leaf curl and stunting',
        'Squash foliage with widespread powdery mildew disease',
        'Close-up of white powdery mildew spores on squash leaf surface'
    ],
    'Strawberry   Leaf scorch': [
        'Strawberry leaves with purple to brown spots from leaf scorch',
        'Strawberry plant showing extensive leaf scorch damage',
        'Close-up of strawberry leaf with characteristic scorch lesions',
        'Severely infected strawberry with leaves turning brown',
        'Young strawberry leaves with early scorch symptoms',
        'Strawberry plant foliage with multiple scorch spots',
        'Mature strawberry leaves with extensive scorching',
        'Strawberry patch with plants affected by leaf scorch',
        'Close-up of strawberry leaf showing purple-bordered scorch lesions',
        'Strawberry plant with premature leaf browning from scorch'
    ],
    'Strawberry   healthy': [
        'Healthy strawberry plant with vibrant green trifoliate leaves',
        'Close-up of unblemished strawberry foliage',
        'Strawberry plant with healthy leaves and fruit development',
        'Pristine strawberry leaves without spots or damage',
        'Young healthy strawberry plant showing vigorous growth',
        'Mature strawberry displaying robust health and berries',
        'Strawberry patch with healthy plants and uniform growth',
        'Close-up of healthy strawberry leaves with proper color',
        'Strawberry plant with lush green foliage and flowers',
        'Healthy strawberry field with optimal plant condition'
    ],
    'Tomato   Bacterial spot': [
        'Tomato leaves with small dark spots from bacterial infection',
        'Tomato fruit showing raised scabby lesions from bacterial spot',
        'Close-up of tomato leaf with numerous bacterial spot symptoms',
        'Severely infected tomato plant with extensive leaf spotting',
        'Young tomato leaves with early bacterial spot lesions',
        'Tomato fruit with dark, raised bacterial spot marks',
        'Tomato plant foliage with widespread bacterial spot damage',
        'Mature tomato leaves with extensive bacterial lesions',
        'Tomato showing deformed growth from bacterial spot',
        'Close-up of tomato leaf with characteristic bacterial spotting pattern'
    ],
    'Tomato   Early blight': [
        'Tomato leaves with circular brown spots showing concentric rings',
        'Tomato plant displaying characteristic target spot lesions from early blight',
        'Close-up of tomato leaf with early blight bull\'s eye pattern',
        'Severely infected tomato with extensive early blight damage',
        'Tomato foliage showing premature yellowing from early blight',
        'Young tomato plant with initial early blight symptoms',
        'Tomato field with plants affected by early blight disease',
        'Mature tomato leaves with large necrotic areas from blight',
        'Tomato plant showing progressive early blight infection',
        'Close-up of tomato leaf with characteristic concentric ring lesions'
    ],
    'Tomato   Late blight': [
        'Tomato leaves with dark, water-soaked lesions from late blight',
        'Tomato plant showing rapid tissue death from late blight',
        'Close-up of tomato leaf with characteristic late blight lesions',
        'Severely infected tomato field with widespread late blight',
        'Tomato foliage with white fungal growth on lesion undersides',
        'Young tomato plant collapsing from late blight infection',
        'Tomato stems showing dark streaks from late blight',
        'Mature tomato leaves with extensive late blight damage',
        'Tomato fruit with brown rotted areas from late blight',
        'Close-up of tomato leaf with water-soaked late blight spots'
    ],
    'Tomato   Leaf Mold': [
        'Tomato leaves with yellow spots on upper surface and olive-green mold underneath',
        'Tomato plant showing extensive leaf mold infection',
        'Close-up of tomato leaf underside with characteristic fuzzy mold growth',
        'Severely infected tomato with leaves curling from leaf mold',
        'Young tomato leaves with early leaf mold symptoms',
        'Tomato plant with leaves showing yellowing and mold patches',
        'Mature tomato foliage with dense leaf mold coverage',
        'Tomato greenhouse with plants affected by leaf mold',
        'Close-up of olive-green to brown mold on tomato leaf underside',
        'Tomato plant with extensive leaf mold causing defoliation'
    ],
    'Tomato   Septoria leaf spot': [
        'Tomato leaves with numerous small circular spots with dark margins',
        'Tomato plant showing extensive septoria leaf spot infection',
        'Close-up of tomato leaf with characteristic spots containing dark pycnidia',
        'Severely infected tomato with premature defoliation from septoria',
        'Young tomato leaves with early septoria leaf spot symptoms',
        'Tomato plant foliage covered with small septoria spots',
        'Mature tomato leaves with extensive septoria damage',
        'Tomato field with plants affected by septoria leaf spot',
        'Close-up of tomato leaf showing gray-centered spots with dark borders',
        'Tomato plant with widespread septoria causing leaf yellowing'
    ],
    'Tomato   Spider mites Two-spotted spider mite': [
        'Tomato leaves with stippling and yellowing from spider mite feeding',
        'Tomato plant showing extensive spider mite damage and webbing',
        'Close-up of tomato leaf underside with spider mites and eggs',
        'Severely infested tomato with leaves turning bronze from mites',
        'Young tomato leaves with early spider mite damage',
        'Tomato plant covered with fine webbing from spider mite infestation',
        'Mature tomato leaves with extensive mite damage and discoloration',
        'Tomato greenhouse with plants affected by spider mites',
        'Close-up of two-spotted spider mites on tomato leaf surface',
        'Tomato plant with leaves showing severe mite feeding damage'
    ],
    'Tomato   Target Spot': [
        'Tomato leaves with circular brown spots showing concentric rings',
        'Tomato plant displaying characteristic target spot lesions',
        'Close-up of tomato leaf with target spot bull\'s eye pattern',
        'Severely infected tomato with extensive target spot damage',
        'Young tomato leaves with early target spot symptoms',
        'Tomato fruit showing sunken lesions from target spot',
        'Tomato plant foliage with widespread target spot disease',
        'Mature tomato leaves with large necrotic areas from target spot',
        'Tomato stems showing target spot lesions',
        'Close-up of tomato leaf with characteristic concentric ring pattern'
    ],
    'Tomato   Tomato Yellow Leaf Curl Virus': [
        'Tomato plant with severely curled and yellowing leaves from TYLCV',
        'Tomato showing stunted growth and upward leaf curling from virus',
        'Close-up of tomato leaf with characteristic yellow curling from TYLCV',
        'Severely infected tomato with extreme leaf curl and reduced size',
        'Young tomato plant with early TYLCV symptoms showing leaf distortion',
        'Tomato plant with bright yellow curled leaves from virus infection',
        'Mature tomato displaying extreme stunting from yellow leaf curl virus',
        'Tomato field with plants affected by TYLCV showing poor fruit set',
        'Close-up of tomato leaves cupping upward with yellow margins',
        'Tomato plant with reduced fruit production due to TYLCV infection'
    ],
    'Tomato   Tomato mosaic virus': [
        'Tomato leaves with mottled light and dark green mosaic pattern',
        'Tomato plant showing leaf distortion from mosaic virus infection',
        'Close-up of tomato leaf with characteristic mosaic discoloration',
        'Severely infected tomato with stunted growth and mosaic symptoms',
        'Young tomato leaves with early mosaic virus patterns',
        'Tomato fruit showing uneven ripening and discoloration from virus',
        'Tomato plant with leaves displaying yellow and green mottling',
        'Mature tomato foliage with extensive mosaic virus symptoms',
        'Tomato showing distorted leaves and reduced vigor from virus',
        'Close-up of tomato leaf with mosaic pattern and leaf narrowing'
    ],
    'Tomato   healthy': [
        'Healthy tomato plant with vibrant green leaves',
        'Close-up of unblemished tomato foliage showing proper color',
        'Tomato plant with healthy leaves and developing fruit',
        'Pristine tomato leaves without spots or lesions',
        'Young healthy tomato plant showing vigorous growth',
        'Mature tomato displaying robust health and fruit production',
        'Tomato crop showing optimal health and yield potential',
        'Close-up of healthy tomato leaves with proper structure',
        'Tomato plant with lush green foliage and flowers',
        'Healthy tomato field with plants at optimal growth stage'
    ]
}

# Simple template for backward compatibility
template = ['a photo of a {}.']


class PlantDisease(DatasetBase):

    dataset_dir = 'Plant_disease'

    def __init__(self, root, num_shots):
        self.dataset_dir = root
        self.num_shots = num_shots

        # Load metadata
        metadata_path = os.path.join(self.dataset_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load classnames
        classnames_path = os.path.join(self.dataset_dir, 'classnames.txt')
        with open(classnames_path, 'r') as f:
            lines = f.readlines()
            self._class_list = [line.strip() for line in lines if line.strip()]

        # Use simple template (修改类名后使用简单模板)
        # 如果要使用详细提示词，需要更新 detailed_prompts 字典的键名与新类名一致
        self.template = simple_template

        # Read train, val, test data
        train = self.read_data(f'train_{num_shots}shot')
        val = self.read_data('val')
        test = self.read_data('test')

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split_dir):
        """Read data from train_Nshot, val, or test directory"""
        split_path = os.path.join(self.dataset_dir, split_dir)
        items = []

        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist")
            return items

        # Iterate through class directories
        class_dirs = sorted(os.listdir(split_path))
        for class_dir in class_dirs:
            class_path = os.path.join(split_path, class_dir)
            if not os.path.isdir(class_path):
                continue

            # Get label and classname
            classname = class_dir.replace('_', ' ')

            # Find label from classnames list
            try:
                label = self._class_list.index(class_dir)
            except ValueError:
                print(f"Warning: {class_dir} not found in classnames.txt")
                continue

            # Read all images in this class directory
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    item = Datum(
                        impath=img_path,
                        label=label,
                        classname=classname
                    )
                    items.append(item)

        print(f"Loaded {len(items)} images from {split_dir}")
        return items
