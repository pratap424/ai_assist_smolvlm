import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

class MiDaSDepth:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "DPT_Large"
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type).to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def estimate_depth(self, pil_image):
        # Ensure image is RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert to numpy and apply transform
        img_np = np.array(pil_image)
        input_tensor = self.transform(img_np).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        # Normalize for display
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8))

        return depth_map, depth_img