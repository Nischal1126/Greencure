from src.model.resnet     import ResNetClassifier
from configs.config       import CONFIG
from src.data.dataloader  import num_classes, train_dataset
import torch
from torchvision import transforms as transform
from PIL import Image



model = ResNetClassifier(
    n_classes  = num_classes,
    pretrained = CONFIG["model"]["pretrained"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "/teamspace/studios/this_studio/checkpoints/best_model.pth"
checkpoint = torch.load(PATH)

config      = CONFIG["training"]

model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()



img = Image.open("/teamspace/studios/this_studio/model_testing/image.png")
print(img.size, img.mode)

preprocess = transform.Compose([
        transform.Resize((224, 224)),
        transform.ToTensor(),
        transform.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.to(device)

with torch.no_grad():
    output = model(img_tensor)

print(f"output logits: {output}")
probabilities = torch.softmax(output, dim=1)
print(f"prob output for all cls: {probabilities}")

_, predicted_idx = torch.max(output, 1)
print(f"output of torch max: {torch.max(output, 1)}")
print(f"predicted output: {predicted_idx}")
print(f"Predicted Class Index: {predicted_idx.item()}")
print(f"predicted class:{train_dataset.classes[predicted_idx.item()]}")
