import matplotlib.pyplot as plt
import argparse
import os 
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from accelerate import Accelerator
import torchvision
import wandb

wandb.login(key="e5b241d7f4f4f5210c5761b5b72436590b6b0767")

### Parse Training Arguments ###
parser = argparse.ArgumentParser(description="Arguments for Image Classification Training")
parser.add_argument("--experiment_name", 
                    help="Name of Experiment being Launched",
                    type=str)
parser.add_argument("--path_to_data", 
                    help="Path to ImageNet root folder which should contain \train and \validation folders", 
                    type=str)
parser.add_argument("--working_directory", 
                    help="Working Directory where checkpoints and logs are stored, inside a \
                    folder labeled by the experiment name",
                    type=str)
parser.add_argument("--epochs",
                    help="Number of Epochs to Train", 
                    type=int)
parser.add_argument("--save_checkpoint_interval", 
                    help="After how many epochs to save model checkpoints",
                    type=int)
parser.add_argument("--num_classes", 
                    help="How many classes is our network predicting?",
                    type=int)
parser.add_argument("--batch_size", 
                    help="Effective batch size. If split_batches is false, batch size is \
                         multiplied by number of GPUs utilize", 
                    type=int)
parser.add_argument("--gradient_accumulation_steps", 
                    help="Number of Gradient Accumulation Steps for Training",  
                    type=int)
parser.add_argument("--weight_decay",
                    help="Weight decay for optimizer",
                    type=float)
parser.add_argument("--learning_rate", 
                    help="Starting Learning Rate for StepLR", 
                    type=float)
parser.add_argument("--num_workers", 
                    help="Number of workers for DataLoader",
                    type=int)
parser.add_argument("--resume_from_checkpoint", 
                    help="Checkpoint folder for model to resume training from, inside the experiment folder", 
                    default=None, 
                    type=str)
args = parser.parse_args()

### Init Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          log_with="wandb")

accelerator.init_trackers(
    project_name="CIFAR10_DDN",  # Must match your WandB project
    config=vars(args)  # Optional: logs all args used in the experiment
)

class SimpleAccuracy:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.correct = 0
        self.total = 0

    def update(self, preds, targets):
        """
        Update accuracy count with new predictions and labels.
        Expects:
        - preds: class indices (batch_size)
        - targets: ground truth labels (batch_size)
        """
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)

    def compute(self):
        """
        Return accuracy as float
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self):
        """
        Clear accumulated stats
        """
        self.correct = 0
        self.total = 0

# Apply any necessary transforms (like converting PIL Image to Tensor if needed later)
transform = transforms.ToTensor()

# Load the dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# Grab a sample image and label
sample, label = trainset[0]  # You can just index directly

# Display it
plt.imshow(np.array(sample))  # sample is a PIL image by default
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()

class CIFAR10(nn.Module):
    def __init__(self, classes=10, dropout_p=0.5):  # CIFAR-10 has 10 classes
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 16 x 16
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 192 x 8 x 8
            nn.BatchNorm2d(192),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 4 x 4
            nn.BatchNorm2d(256),
        )

        self.head = nn.Sequential(
            nn.Flatten(),  # Flatten to shape: (256*4*4)
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x
    
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
'''
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),
])

#Keeping the same for the training phase
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

'''
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

# Load datasets
trainset = torchvision.datasets.CIFAR10(
    root=args.path_to_data,
    train=True,
    download=True,
    transform=train_transforms
)

testset = torchvision.datasets.CIFAR10(
    root=args.path_to_data,
    train=False,
    download=True,
    transform=test_transform
)

# Set batch size
BATCH_SIZE = 64  # You can change this to 32, 128, etc., depending on your system
mini_batchsize = args.batch_size // args.gradient_accumulation_steps

# Create data loaders
trainloader = DataLoader(trainset, batch_size=mini_batchsize, shuffle=True, num_workers = 2)
testloader = DataLoader(testset, batch_size=mini_batchsize, shuffle=False, num_workers = 2)

device = "cuda" if torch.cuda.is_available() else "cpu"

### Set the Loss Function ###
loss_fn = nn.CrossEntropyLoss()

### Load the Model ###
model = CIFAR10(classes=10).to(accelerator.device)

### Set the Optimizer ###
learning_rate = .001
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

###MORE EFFECTIVE
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
    model, optimizer, trainloader, testloader, scheduler
)

accelerator.register_for_checkpointing(scheduler)

if args.resume_from_checkpoint is not None:
    accelerator.print(f"Resuming from Checkpoint: {args.resume_from_checkpoint}")
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    accelerator.load_state(path_to_checkpoint)
    starting_epoch = int(args.resume_from_checkpoint.split("_")[-1])
else:
    starting_epoch = 0

###accuracy_fn = Accuracy(task="multiclass", num_classes=args.num_classes).to(accelerator.device)
accuracy_fn = SimpleAccuracy(num_classes=args.num_classes, device=accelerator.device)
# Store metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(starting_epoch, (args.epochs + 1)):

    accelerator.print(f"Training Epoch {epoch}")
    
    # Reset accuracy tracker at the beginning of each epoch
    accuracy_fn.reset()

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    
    accumulated_loss = 0
    accumulated_accuracy = 0

    progress_bar = tqdm(range(len(trainloader) // args.gradient_accumulation_steps),
                         disable=not accelerator.is_main_process)
    ### TRAINING ####
    model.train()
    for images, targets in trainloader:
        images = images.to(accelerator.device)
        targets = targets.to(accelerator.device)

        with accelerator.accumulate(model):
            pred = model(images)

            loss = loss_fn(pred, targets)
            accumulated_loss += loss / args.gradient_accumulation_steps

            predicted = pred.argmax(axis=-1)
            accuracy_fn.update(predicted, targets)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

    # Only append averaged loss after gradient accumulation step
            if accelerator.sync_gradients:
                loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
                train_loss.append(torch.mean(loss_gathered).item())  # Append the averaged loss
        accumulated_loss = 0  # Reset the accumulated loss for the next batch

    train_acc.append(accuracy_fn.compute())
    accuracy_fn.reset()
    progress_bar.update(1)

    ### TESTING ###
    model.eval()
    for images, targets in tqdm(testloader, disable=not accelerator.is_local_main_process):
        images = images.to(accelerator.device)
        targets = targets.to(accelerator.device)
        
        with torch.no_grad():
            pred = model(images)
            loss = loss_fn(pred, targets)
        predicted = pred.argmax(axis=-1)
        ###accuracy = accuracy_fn(predicted, targets)  # Update accuracy
        accuracy_fn.update(predicted, targets)
        loss_gathered = accelerator.gather_for_metrics(loss)
        ###accuracy_gathered = accelerator.gather_for_metrics(accuracy)

        test_loss.append(torch.mean(loss_gathered).item())
    test_acc.append(accuracy_fn.compute())
    accuracy_fn.reset()  # Compute accuracy at the end of the epoch

    epoch_train_loss = np.mean(train_loss)
    epoch_test_loss = np.mean(test_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_test_acc = np.mean(test_acc)

    accelerator.print(f"Training Accuracy: {epoch_train_acc}, Training Loss: {epoch_train_loss}")
    accelerator.print(f"Testing Accuracy: {epoch_test_acc}, Testing Loss: {epoch_test_loss}")

    ### LOG WITH WEIGHTS AND BIASES
    accelerator.log({"learning_rate": scheduler.get_last_lr()[0],
                     "training_loss": epoch_train_loss,
                     "testing_loss": epoch_test_loss,
                     "training_acc": epoch_train_acc,
                     "testing_acc": epoch_test_acc}, step=epoch)

    ### ITERATE LR SCHEDULER
    scheduler.step(epoch)

    if epoch % args.save_checkpoint_interval == 0:
        path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{epoch}")
        accelerator.save_state(output_dir=path_to_checkpoint)

accelerator.end_training()


'''
    ### TRAINING ####
    model.train()
    for images, targets in trainloader:
        images = images.to(accelerator.device)
        targets = targets.to(accelerator.device)

        with accelerator.accumulate(model):

            pred = model(images)

            loss = loss_fn(pred, targets)
            accumulated_loss += loss / args.gradient_accumulation_steps

            predicted = pred.argmax(axis=-1)
            accuracy_fn.update(predicted, targets)
            ##accuracy= accuracy_fn(predicted, targets)  # Update accuracy
            ##accumulated_accuracy += accuracy / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
            #accuracy_gathered = accelerator.gather_for_metrics(accumulated_accuracy)
                train_loss.append(torch.mean(loss_gathered).item())
    train_acc.append(accuracy_fn.compute())
    accuracy_fn.reset()
            ###train_acc.append(torch.mean(accuracy_gathered).item())  # Compute accuracy at the end of the epoch
for epoch in range(epochs):
    #Training Phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(trainloader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    #Validation Phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    val_loss = val_loss / len(testloader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
### First time running drop out rate set to .1 
    #This led to the best epoch at 9 
    #training acc 95.7% 
    #val acc 81.59%
    #Still seeing overfitting issues 
    ## Fixing the drop out to .3 and seeing if results get better.
    ### Drop out rate at .3
        #Best epoch at 6
        #training acc 91.23%
        #val acc 81.51%
        #roughly same results that dropout of .1 had 
        #overfitting issues
    ### Drop out rate at .5
        #produced the best results
        #At epoch 10
        # training acc 95.51%
        # val acc 83.05%
        #Overfitting is still evident, but not as much as before
### Now introducing randomness by utilizing horizontal flip / random crop to the transform
    ###This should help with the issue of overfitting.
    ###Best results
    ###Training and Testing are now producing values in line with eachother
    ###epoch 10
        # Training acc 86.90%
        # Val acc 84.60%
    ###Now going to run more epochs (20)
        ##Best epoch at 18
            #Training acc 91.67%
            #Val acc 88.22%
### No more overfitting with model 
### Val acc levels out around epoch 15
'''
