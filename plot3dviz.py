import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from visualization import ANN

model = ANN.load_from_checkpoint(
    r".\lightning_logs\version_5\checkpoints\epoch=4-step=8594.ckpt")
model = model.cuda()
model = model.eval()
transform = transforms.Compose([transforms.ToTensor()])
data = MNIST(".", train=False, download=True, transform=transform)
np.random.seed(19)
num_wrong_samples = 1000
random_indices = np.random.uniform(0, 10000,
                                   num_wrong_samples).astype(np.uint8)
outlier_list = []
for i in range(num_wrong_samples):
    outlier = np.random.uniform(0, 255, (28, 28)).astype(np.uint8)
    outlier_list.append(outlier)
for idx in range(num_wrong_samples):
    data.data[random_indices[idx]] = torch.ByteTensor(outlier_list[idx])
    data.targets[random_indices[idx]] = 10
dataloader = DataLoader(data, batch_size=32)

test_imgs = torch.zeros((0, 1, 28, 28), dtype=torch.float32)
test_predictions = []
test_targets = []
test_embeddings = torch.zeros((0, 100), dtype=torch.float32)
for x, y in dataloader:
    x = x.cuda()
    embeddings, logits = model(x)
    preds = torch.argmax(logits, dim=1)
    test_predictions.extend(preds.detach().cpu().tolist())
    test_targets.extend(y.detach().cpu().tolist())
    test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()),
                                0)
    test_imgs = torch.cat((test_imgs, x.detach().cpu()), 0)
test_imgs = np.array(test_imgs)
test_embeddings = np.array(test_embeddings)
test_targets = np.array(test_targets)
test_predictions = np.array(test_predictions)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

tsne = TSNE(3, verbose=1)
tsne_proj = tsne.fit_transform(test_embeddings)
cmap = cm.get_cmap('tab20')
num_categories = 10
for lab in range(num_categories):
    indices = test_predictions == lab
    ax.scatter(tsne_proj[indices, 0],
               tsne_proj[indices, 1],
               tsne_proj[indices, 2],
               c=np.array(cmap(lab)).reshape(1, 4),
               label=lab,
               alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.show()

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
pca = PCA(n_components=3)
pca.fit(test_embeddings)
pca_proj = pca.transform(test_embeddings)

num_categories = 10
for lab in range(num_categories):
    indices = test_predictions == lab
    ax.scatter(pca_proj[indices, 0],
               pca_proj[indices, 1],
               pca_proj[indices, 2],
               c=np.array(cmap(lab)).reshape(1, 4),
               label=lab,
               alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.show()
