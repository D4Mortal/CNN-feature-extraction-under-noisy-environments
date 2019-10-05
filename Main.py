# Applying 10-fold cross validation on SFEW dataset
#
# program written by Daniel Hao, May 2019


from torchvision import datasets, transforms
from CNN_Functions import k_fold_crossvalidation
    
root_path = "Subset For SFEW"

transform = transforms.Compose([
         transforms.Resize([360, 288]),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],
                         std=[1, 1, 1])
         ])

data_all = datasets.ImageFolder(root = root_path, transform = transform)

k_fold_crossvalidation(10, data_all, 12)