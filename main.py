import os
import shutil
import argparse
import module as m
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
from datetime import datetime
import numpy as np


def main(cli_args: argparse.Namespace) -> None:
    data_dir = "./data"

    # guarantee that the current working directory is the directory that main is in
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # guarantee that destination is within ./model, if not, then put it in ./model
    if not cli_args.dest.startswith("./model"):
        cli_args.dest = f"./model/{cli_args.dest}"

    # create destination directory if it does not exist
    os.makedirs(cli_args.dest, exist_ok=True)

    # If destination already exists, then move all contents to backup
    if os.listdir(cli_args.dest):
        count = 0
        new_dest_dir = f"{cli_args.dest}_backup_{count}"
        while os.path.exists(new_dest_dir) and os.listdir(new_dest_dir):
            count += 1
            new_dest_dir = f"{cli_args.dest}_backup_{count}"

        shutil.move(cli_args.dest, new_dest_dir)

    # load configuration data
    args = m.load_config()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    train_loader, test_loader = m.load_dataset(
        root=data_dir,
        name=args["data"]["name"],
        batch_size=args["data"]["batch_size"],
        num_workers=args["data"]["num_workers"],
        padding=args["data"]["padding"],
    )

    # get dataset properties
    classes: list[str] = train_loader.dataset.classes
    dims: tuple[int, int] = train_loader.dataset[0][0].shape

    # load model
    model = m.get_model(
        name=args["model"]["name"], depth=dims[0], num_classes=len(classes)
    )
    model.to(device)

    # get criterion, optimizer, and scheduler
    criterion = m.get_module(torch.nn, args["criterion"]["name"])
    optimizer = m.get_optimizer(
        model, args["optimizer"]["name"], **args["optimizer"]["args"]
    )
    scheduler = m.get_scheduler(
        optimizer, args["scheduler"]["name"], **args["scheduler"]["args"]
    )

    # Create a TensorBoard SummaryWriter and Display dataset images data
    dataiter = iter(train_loader)
    images, _ = next(dataiter)
    img_grid = torchvision.utils.make_grid(images)

    # Add model graph to TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(cli_args.dest, args["files"]["log"]))
    writer.add_image("64 CIFAR100 Images", img_grid)
    writer.add_graph(model, images.to(device))

    # Training loop
    best_acc = 0
    epochs = args["model"]["epochs"]
    interval = args["model"]["save_interval"]
    train_losses, train_accs = np.zeros(epochs), np.zeros(epochs)
    test_losses, test_accs = np.zeros(epochs), np.zeros(epochs)

    for epoch in m.ProgressBar(range(epochs), title="Training"):
        # Train
        train_loss, train_acc = m.train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        train_losses[epoch] = train_loss
        train_accs[epoch] = train_acc

        # Evaluate
        test_loss, test_acc, all_preds, all_labels = m.evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )
        test_losses[epoch] = test_loss
        test_accs[epoch] = test_acc

        # Tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        # Save model at checkpoints
        if epoch % args["model"]["save_interval"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "preds": all_preds,
                    "labels": all_labels,
                    "test_acc": test_acc,
                    "config": args,
                },
                f"{cli_args.dest}/model_{epoch//interval}.pth",
            )
            m.ProgressBar.write(
                f"Epoch: {epoch} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
            )

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "preds": all_preds,
                    "labels": all_labels,
                    "test_acc": test_acc,
                    "config": args,
                },
                f"{cli_args.dest}/model_best.pth",
            )

        # Adjust learning rate
        if scheduler.__class__.__name__ == "ReduceLROnPlateau":
            scheduler.step(test_loss)
        else:
            scheduler.step()

    # Visualize model
    # TODO ADD VISUALS
    model_name = f"{cli_args.dest}/model_best.pth"

    # Close TensorBoard
    writer.flush()
    writer.close()


if __name__ == "__main__":
    # get CLI arguments
    parser = argparse.ArgumentParser(
        description="Train a model with the given configuration."
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=f"model_{datetime.now().strftime("%Y-%m-%d_%Hh%Mm")}",
        help="Directory to save the output.",
    )
    cli_args = parser.parse_args()

    # entrypoint
    main(cli_args)
