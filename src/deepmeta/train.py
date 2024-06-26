"""modfied training script from https://github.com/cbib/DeepMeta """
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import progressbar
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils import data as tud

from src.deepmeta.utils import data
from src.deepmeta.utils import pprint
from src.deepmeta.utils import utils

# widget list for the progress bar
widgets = [
    " [",
    progressbar.Timer(),
    "] ",
    progressbar.Bar(),
    " (",
    progressbar.ETA(),
    ") ",
]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
LOSS = np.inf
METRIC = np.array([-1.0, -1.0, -1.0])

def get_args() -> argparse.Namespace:
    """
    Argument parser.

    :return: Object containing all the parameters needed to train a model
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=64, help="size of the batches"
    )
    parser.add_argument("--drop", "-d", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "-nworkers", type=int, default=4, help="Number of workers in dataloader."
    )
    parser.add_argument(
        "-classes", type=int, default=3, help="Number of classes to predict."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="iou",
        help="Metric for stats. iou or f1",
        choices=["iou", "f1", "auc"],
    )
    parser.set_defaults(save=False)
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        help="Model name. unet or unet_res",
        choices=[
            "unet",
            "unet3p",
            "deepmeta",
            "vanilla_unet",
        ],
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        help="Loss function.",
        choices=["ce", "weighted_ce",'unetpp_loss',"focal", "lovasz", "fusion", 'unet3p_loss'],
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the data dir."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='results',
        help="Path to dir where saving models and output."
    )
    parser.add_argument(
        "--dim",
        type=str,
        default="2d",
        help="dimension to choose for training",
        choices=[
            "2d",
            "3d"
        ],
    )
    args = parser.parse_args()
    return args

def save_checkpoint(model:nn.Module,
                    optimizer:torch.optim.Optimizer,
                    epoch:int,
                    filepath:str) -> None:
    """saves checkpoint for a particular epoch
    Args:
        model(nn.Module) - the model to be saved
        optimizer(torch.optim.Optimizer) - optimizer state
        epoch (int) - epoch no.
        filepath (str) - where to save the checkpoint
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filepath)
    print(f"Checkpoint saved at epoch {epoch}")

def save_model(net: nn.Module,
               loss: float,
               save_path:str) -> None:
    """
    Save the model if the loss is lower than the previous one.

    :param net: The model to save
    :type net: nn.Module
    :param loss: Loss value of the model during the validation step
    :type loss: float
    """
    global LOSS
    if loss < LOSS:
        LOSS = loss
        torch.save(net.state_dict(), save_path)
        pprint.print_bold_red("Model saved")

def _step(
    net: nn.Module,
    dataloader: Dict[str, tud.DataLoader],
    args: argparse.Namespace,
    step: str,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str = "cuda"
) -> float:
    """
    Train or validate the network

    :param net: The network to train
    :type net: nn.Module
    :param dataloader: The dataloader for the training and validation sets
    :type dataloader: Dict[str, tud.DataLoader]
    :param args: The arguments from the command line
    :type args: argparse.Namespace
    :param step: The step to train or validate
    :type step: str
    :param optimizer: The optimizer to use
    :type optimizer: torch.optim.Optimizer
    :return: The loss of the step
    :rtype: float
    """
    if 3==args.classes:
        metric_labels = [0,1,2]
    else:
        metric_labels = [0,1]
    criterion = utils.get_loss(args,
                               device=device,
                               nclass=args.classes,
                               dim=args.dim)
    running_loss = []
    net.train() if step == "Train" else net.eval()
    dataset = dataloader[step]
    with progressbar.ProgressBar(max_value=len(dataset), widgets=widgets) as bar:
        f1 = []
        for i, (inputs, labels) in enumerate(dataset):
            bar.update(i)
            inputs, labels = inputs.to(device, non_blocking=True), (labels.long()).to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels.squeeze(1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss.append(loss.item())
            outputs = outputs.max(1).indices
            labels = labels.squeeze(1)
            f1.append(
                utils.get_metric(args.metric)(
                    torch.flatten(labels).cpu().detach().numpy(),
                    torch.flatten(outputs).cpu().detach().numpy(),
                    average=None,
                    labels=metric_labels,
                    zero_division=1,
                )
            )
    f1_metric = np.array(f1).mean(0)
    epoch_loss = np.array(running_loss).mean()
    if 3==args.classes:
        pprint.print_mag(
            f"[{step}] loss: {epoch_loss:.3f} {args.metric} bg: {f1_metric[0]:.5f}  "
            f"{args.metric} lungs: {f1_metric[1]:.5f} {args.metric} metas: {f1_metric[2]:.5f}"  # noqa
    )
    else:
        pprint.print_mag(
            f"[{step}] loss: {epoch_loss:.3f} {args.metric} bg: {f1_metric[0]:.5f}  "
            f"{args.metric} metas: {f1_metric[1]:.5f}"  # noqa
    )
    if step == "Val":
        pathway=os.path.join(args.save_dir, f"models/best_model_{args.model}_{args.dim}.pth")
        save_model(net, epoch_loss, pathway)
    return epoch_loss


def train(
    net: nn.Module,
    dataloader: Dict[str, tud.DataLoader],
    args: argparse.Namespace
) -> Tuple[List[float], List[float], nn.Module]:
    """
    Train the network

    Parameters
    ----------
    net : nn.Module
        The network to train
    dataloader : Dict[str, tud.DataLoader]
        The dataloader for the training and validation sets
    args : argparse.Namespace
        The arguments from the command line
    """
    os.makedirs(os.path.join(args.save_dir,'models'), exist_ok=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 5, 1
    )
    history_train, history_val = [], []
    print(100 * "-")
    print("Start training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} :")
        for step in ["Train", "Val"]:
            epoch_loss = _step(
                net, dataloader, args, step, optimizer, scaler,
            )
            if step == "Val":
                history_val.append(epoch_loss)
            else:
                history_train.append(epoch_loss)
        scheduler.step()
        save_checkpoint(net, optimizer, epoch, f'checkpoint_{args.model}_{args.loss}_{args.dim}.pth')
    pprint.print_bold_green("Finished Training")
    pathway=os.path.join(args.save_dir, f"models/final_model_{args.model}_{args.dim}.pth")
    torch.save(net.state_dict(), pathway)
    return history_train, history_val, net

def plot_learning_curves(
    hist_train: list,
    hist_val: list,
    path: str = "results/"
) -> None:
    """
    Plot training curves.

    :param hist_train: List of epoch loss value for training
    :type hist_train: list
    :param hist_val: List of epoch loss value for validation
    :type hist_val: list
    :param path: Saving path
    :type path: str

    """
    pathway = os.path.join(path, 'plots')
    os.makedirs(pathway, exist_ok=True)
    epochs = range(len(hist_train))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, hist_train, label="Train loss")
    plt.plot(epochs, hist_val, label="Validation loss")
    plt.title("Loss - Training vs. Validation.")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    config = get_args()
    pathway = os.path.join(pathway, f"training_curves_{config.model}_{config.loss}_{config.dim}.png")
    plt.savefig(pathway)

def main():
    config = get_args()
    print(config.model)
    torch.cuda.empty_cache()
    model = utils.get_model(config).cuda()
    if config.dim=='3d':
        model=nn.DataParallel(model)
    dataloader = data.get_datasets(os.path.join(config.data_path,"images/"),
                                   os.path.join(config.data_path, "labels/"),
                                   config)
    log_train, log_val, _ = train(model, dataloader, config)
    plot_learning_curves(log_train, log_val, config.save_dir)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
