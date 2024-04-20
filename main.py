import torch

from model import DetectionModel


def main():
    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    model = DetectionModel("n", 80)
    print(f"total parameter: {count_parameters(model)}")

    model.load_state_dict(torch.load(r"best_state_dict.pt"))
    model.eval()
    img = torch.ones(1, 3, 640, 640)
    out = model(img)
    print(out.shape)


if __name__ == "__main__":
    main()