import torch
import torch.utils.data
from torchvision import transforms

from data import Dataset, Sampler
import transporter
import utils


def get_config():
    config = utils.ConfigDict({})
    config.dataset_root = 'data'
    config.batch_size = 64
    config.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    config.image_channels = 3
    config.k = 4
    config.num_iterations = int(1e6)
    config.learning_rate = 1e-3
    config.learning_rate_decay_rate = 0.95
    config.learning_rate_decay_every_n_steps = int(1e5)
    config.report_every_n_steps = 100

    return config


def _get_model(config):
    feature_encoder = transporter.FeatureEncoder(config.image_channels)
    pose_regressor = transporter.PoseRegressor(config.image_channels, config.k)
    refine_net = transporter.RefineNet(config.image_channels)

    return transporter.Transporter(feature_encoder, pose_regressor, refine_net)

def _get_data_loader(config):
    transform = transforms.ToTensor()
    dataset = Dataset(config.dataset_root, transform=transform)
    sampler = Sampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, sampler=sampler, pin_memory=True, num_workers=4)
    return loader


def main():
    config = get_config()

    model = _get_model(config)
    model = model.to(config.device)

    loader = _get_data_loader(config)

    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        config.learning_rate_decay_every_n_steps,
        gamma=config.learning_rate_decay_rate)

    for itr, (xt, xtp1) in enumerate(loader):
        model.train()
        xt = xt.to(config.device)
        xtp1 = xtp1.to(config.device)
        optimizer.zero_grad()
        reconstruction = model(xt, xtp1)
        loss = torch.nn.functional.mse_loss(reconstruction, xtp1)
        loss.backward()

        optimizer.step()
        scheduler.step()
        if itr % config.report_every_n_steps == 0:
            print("Itr ", itr, "Loss ", loss)
            torch.save(model.state_dict(), "model.pth")
        if itr > config.num_iterations:
            break


if __name__ == "__main__":
    main()
