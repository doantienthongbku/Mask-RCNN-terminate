
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


class ResNet50_extract_feature(nn.Module):
    """Extract feature from ResNet50
    """
    def __init__(self):
        super(ResNet50_extract_feature, self).__init__()

        net = models.ResNet50(pretrained=True)
        # create feature extractor from ResNet50 network
        self.body = create_feature_extractor(
            net, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        
    def forward(self, x):
        x = self.body(x)
        return x

    def stages(self):
        return self.body
