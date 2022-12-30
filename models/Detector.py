import torch
from PIL import Image
from torchvision import transforms
from models.cct import cct_14_7x2_384_fl

class Detector():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mean = [0.4353, 0.3773, 0.2872]
        std = [0.2966, 0.2455, 0.2698]
        img_size = 384
        self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.model = cct_14_7x2_384_fl(pretrained=True, progress=True).to(self.device)
        self.flower_name = {'21': 'fire lily',
 '3': 'canterbury bells',
 '45': 'bolero deep blue',
 '1': 'pink primrose',
 '34': 'mexican aster',
 '27': 'prince of wales feathers',
 '7': 'moon orchid',
 '16': 'globe-flower',
 '25': 'grape hyacinth',
 '26': 'corn poppy',
 '79': 'toad lily',
 '39': 'siam tulip',
 '24': 'red ginger',
 '67': 'spring crocus',
 '35': 'alpine sea holly',
 '32': 'garden phlox',
 '10': 'globe thistle',
 '6': 'tiger lily',
 '93': 'ball moss',
 '33': 'love in the mist',
 '9': 'monkshood',
 '102': 'blackberry lily',
 '14': 'spear thistle',
 '19': 'balloon flower',
 '100': 'blanket flower',
 '13': 'king protea',
 '49': 'oxeye daisy',
 '15': 'yellow iris',
 '61': 'cautleya spicata',
 '31': 'carnation',
 '64': 'silverbush',
 '68': 'bearded iris',
 '63': 'black-eyed susan',
 '69': 'windflower',
 '62': 'japanese anemone',
 '20': 'giant white arum lily',
 '38': 'great masterwort',
 '4': 'sweet pea',
 '86': 'tree mallow',
 '101': 'trumpet creeper',
 '42': 'daffodil',
 '22': 'pincushion flower',
 '2': 'hard-leaved pocket orchid',
 '54': 'sunflower',
 '66': 'osteospermum',
 '70': 'tree poppy',
 '85': 'desert-rose',
 '99': 'bromelia',
 '87': 'magnolia',
 '5': 'english marigold',
 '92': 'bee balm',
 '28': 'stemless gentian',
 '97': 'mallow',
 '57': 'gaura',
 '40': 'lenten rose',
 '47': 'marigold',
 '59': 'orange dahlia',
 '48': 'buttercup',
 '55': 'pelargonium',
 '36': 'ruby-lipped cattleya',
 '91': 'hippeastrum',
 '29': 'artichoke',
 '71': 'gazania',
 '90': 'canna lily',
 '18': 'peruvian lily',
 '98': 'mexican petunia',
 '8': 'bird of paradise',
 '30': 'sweet william',
 '17': 'purple coneflower',
 '52': 'wild pansy',
 '84': 'columbine',
 '12': "colt's foot",
 '11': 'snapdragon',
 '96': 'camellia',
 '23': 'fritillary',
 '50': 'common dandelion',
 '44': 'poinsettia',
 '53': 'primula',
 '72': 'azalea',
 '65': 'californian poppy',
 '80': 'anthurium',
 '76': 'morning glory',
 '37': 'cape flower',
 '56': 'bishop of llandaff',
 '60': 'pink-yellow dahlia',
 '82': 'clematis',
 '58': 'geranium',
 '75': 'thorn apple',
 '41': 'barbeton daisy',
 '95': 'bougainvillea',
 '43': 'sword lily',
 '83': 'hibiscus',
 '78': 'lotus lotus',
 '88': 'cyclamen',
 '94': 'foxglove',
 '81': 'frangipani',
 '74': 'rose',
 '89': 'watercress',
 '73': 'water lily',
 '46': 'wallflower',
 '77': 'passion flower',
 '51': 'petunia'}


    def forward(self, filename):
        img = Image.open(filename)
        x = self.transform(img)
        x = x.unsqueeze(0)
        x = x.to(self.device)
        out = self.model(x)
        pred = out.max(1, keepdim=True)[1].item()
        return self.flower_name[str(pred)]

    def ROI(self, src, resize_parameter=0.5):
        '''
        # 自动获取中间的ROI区域
        :param src: 输入图片
        :param resize_parameter: 中心ROI的大小
        :return:
        '''
        height, width, layer = src.shape
        # 图片中心x_center=(x_c,y_c)
        x_c, y_c = width / 2, height / 2

        # ROI的长宽占图像的resize_parameter倍
        w = int(width * resize_parameter)
        h = int(height * resize_parameter)

        x_min = int(x_c - w / 2)
        y_min = int(y_c - h / 2)

        # roi区域 r=(x_min, y_min, w, h)
        roi = src[y_min:y_min + h, x_min:x_min + w]

        return roi


    def predict(self, filename):
        flower_name = self.forward(filename)
        describe = 'This is a flower'

        return flower_name, describe