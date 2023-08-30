import numpy as np
import torch.nn as nn
import torchreid
import cv2
from PIL import Image
import numpy as np
from torchvision.ops import RoIPool
from torchvision.transforms import ToTensor
import torch


# Rebuild OSNet to return a feature map size (1, 512, 32, 16)
class OSNet_x1_0(nn.Module):
    def __init__(self):
        super(OSNet_x1_0, self).__init__()
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1041,
            loss='softmax',
            pretrained=False
        )
        weight_path = 'osnet_ms_d_c.pth.tar'
        torchreid.utils.load_pretrained_weights(model, weight_path)
        self.conv1 = model.conv1  # Skip MaxPooling2D layer between conv1 and conv2 to keep output size at 32*16*512
        self.conv2 = model.conv2
        self.conv3 = model.conv3
        self.conv4 = model.conv4
        self.conv5 = model.conv5

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class FeatureExtractor:
    def __init__(self) -> None:
        self.device = torch.device("cuda")
        self.model = self.loadModel()

    def loadModel(self):
        osnet = OSNet_x1_0()
        osnet.to(self.device)
        osnet.eval()
        return osnet

    # Using RoI Pooling to return a fixed size feature map size [512,7, 7] and Average Pooling to return feature map size [512, 1, 1]
    def roi(self, ft_map, bbox):
        roi_pooling = RoIPool((7, 7), spatial_scale=1.0).to(self.device)
        out = roi_pooling(ft_map, bbox)
        avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        pool_feature = avg_pooling(out)
        app = pool_feature.view(pool_feature.size()[0], pool_feature.size()[1])
        return app

    def extract(self, frame: np.ndarray, boxes: list) -> list[np.ndarray]:
        """
        frame: BGR image, with shape (1080, 1920, 3), dtype = uint8
        boxes: list of bounding boxes.
            [
                (l, t, r, b),
                (l, t, r, b),
                ...
            ]
            where (l, t, r, b) are (left, top, right, bottom) of each bounding box
        :return: list of feature vectors, corresponding to each bounding box
        Each feature vector is a np ndarray with shape = (512, )
        """
        # Convert bgr numpy image to rgb tensor
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # frame BGR image -> RGB image
        pil_image = Image.fromarray(rgb_image)  # Create Image object from numpy.ndarray
        img_rsz = pil_image.resize((256, 128), Image.ANTIALIAS)

        # Extract feature from whole image
        img = ToTensor()(img_rsz)
        img = img.to(self.device)
        img = img.view(-1, img.size(0), img.size(1), img.size(2))
        feature_map = self.model(img)

        # Scaling bounding boxes to feature map's size
        scale_x = pil_image.size[0] / feature_map.size(3)
        scale_y = pil_image.size[1] / feature_map.size(2)

        e = 0.50001  # Prevent 0 width & height in feature map
        bbx = [[0, min(l / scale_x, feature_map.size(3) - e), min(t / scale_y, feature_map.size(2) - e),
                r / scale_x, b / scale_y] for l, t, r, b in boxes]
        bbx = torch.tensor(bbx).to(self.device)
        feature_vectors = self.roi(feature_map, bbx)
        result = [tensor.detach().cpu().squeeze().numpy() for tensor in
                  feature_vectors]  # convert from torch.Tensor to numpy.ndarray

        return result


if __name__ == '__main__':
    # img_path = '/content/drive/MyDrive/000001.jpg'
    # bgr_image = cv2.imread(img_path)
    # bbx = [(100, 200, 300, 400), [100, 200, 301, 401], [101, 201, 300, 400]]
    feature_extractor = FeatureExtractor()
    feature_extractor.extract(frame, boxes)
