from picamera import PiCamera
from time import sleep
import glob


class PyronearEngine:
    """This class is the Pyronear Engine. This engine manage the whole Fire Detection
       process by capturing and saving the image and prediction if there is a fire or
       not based on this image.
    Example
    -------
    pyronearEngine = PyronearEngine()
    pyronearEngine.run(30) # For a prediction every 30s
    """
    def __init__(self, imgsFolder):
        # Camera
        self.camera = PiCamera()

        # Images Folder
        self.imgsFolder = imgsFolder

        # Model definition
        self.model = torchvision.models.resnet18(pretrained=True)

        # Change fc
        in_features = getattr(self.model, 'fc').in_features
        setattr(self.model, 'fc', nn.Linear(in_features, 2))

        self.model.load_state_dict(torch.load("model/model_resnet18.txt"))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.tf = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      normalize])

    def run(self, timeStep):

        imgs = glob.glob(self.imgsFolder)
        idx = len(imgs)

        while True:
            imagePath = self.imgsFolder + "//" + str(idx).zfill(8) + ".jpg"

            self.capture(imagePath)
            pred = self.predict(imagePath)
            print(pred)

            sleep(timeStep)

    def capture(self, imagePath):

        self.camera.start_preview()
        sleep(3)  # gives the camera’s sensor time to sense the light levels
        camera.capture(imagePath)
        camera.stop_preview()

    def predict(self, imagePath):
        im = Image.open(imPath)
        imT = self.tf(im)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(imT.unsqueeze(0))

        if pred[0, 0] > pred[0, 1]:
            return "no fire"
        else:
            return "FIRE !!!"
