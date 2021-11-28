from google_drive_downloader import GoogleDriveDownloader as gdd
from pyrovision.models.rexnet import rexnet1_0x
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import json


# Download dataset
gdd.download_file_from_google_drive(file_id='13-5NJuVr29m0n3MG9wcYLjdgu2_2Hye0',
                                    dest_path='./ds.zip',
                                    unzip=True)


# Init
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

tf = transforms.Compose([transforms.Resize(size=(448)), transforms.CenterCrop(size=448),
                         transforms.ToTensor(), normalize])

model = rexnet1_0x(pretrained=True).eval()

ds_val = ImageFolder(root='ardeche_test_ds', transform=tf)
val_loader = DataLoader(ds_val, batch_size=4)

tp, tn, num_samples_fire, num_samples_no_fire = 0, 0, 0, 0
for x, target in val_loader:

    # Forward
    out = model(x)

    # Apply sigmoid
    out = torch.sigmoid(out)

    samples_fire = target >= 0.4
    samples_no_fire = target < 0.4

    num_samples_fire += int(torch.sum(samples_fire).item())
    num_samples_no_fire += int(torch.sum(samples_no_fire).item())

    tp += int(torch.sum(out[samples_fire] >= 0.4).item())
    tn += int(torch.sum(out[samples_no_fire] < 0.4).item())


fp = num_samples_fire - tp
fn = num_samples_no_fire - tn
accuracy = (tp + tn) / (tp + fp + tn + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)

# Write results to file
with open("references/classification/test_score.json", 'w') as outfile:
    json.dump({"accuracy": accuracy, "recall": recall, "precision": precision}, outfile)
