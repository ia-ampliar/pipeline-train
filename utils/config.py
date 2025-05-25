import torch
from torch.utils.tensorboard import SummaryWriter

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def get_callbacks(log_dir):
    """Retorna o SummaryWriter para TensorBoard"""
    writer = SummaryWriter(log_dir)
    return writer


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint salvo em {path}")
