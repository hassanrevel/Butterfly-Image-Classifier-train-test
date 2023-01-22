import os
import torch
from tqdm.auto import tqdm
import argparse
from src.model import efficientmodel
from src.datasets import get_data_loader
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str,
                        help='Path where the model weights has been stored', required=False)
    parser.add_argument('--data_dir', type=str,
                        help="path of directory where the data has been stored", required=True)
    parser.add_argument('--batch_size', type=int, default=16,
                        required=False)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_classes = len(os.listdir(os.path.join(args.data_dir, 'train')))

    model = efficientmodel(pretrained=False, fine_tune=False, num_classes=num_classes).to(device)
    checkpoint = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    ## data
    _, _, test_loader = get_data_loader(args, use_cuda)

    print(f"Computation device: {device}")

    ## Running the testing script
    pred_list = torch.zeros(0, dtype=torch.long, device=device)
    label_list = torch.zeros(0, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for _, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)

            _, preds = torch.max(logits.data, 1)

            pred_list = torch.cat([pred_list, preds.view(-1).to(device)])
            label_list = torch.cat([label_list, labels.view(-1).to(device)])

    accuracy = int(multiclass_accuracy(pred_list, label_list, num_classes=num_classes) * 100)
    precision = int(multiclass_precision(pred_list, label_list, num_classes=num_classes) * 100)
    recall = int(multiclass_recall(pred_list, label_list, num_classes=num_classes) * 100)
    f1_score = int(multiclass_f1_score(pred_list, label_list, num_classes=num_classes) * 100)

    print("================ Testing Results =====================")

    print(f"Acccuracy         |      {accuracy}%")
    print(f"Precision         |      {precision}%")
    print(f"Recall            |      {recall}%")
    print(f"f1 Score          |      {f1_score}%")

    print("==============================================")

if __name__ == "__main__":
    main()