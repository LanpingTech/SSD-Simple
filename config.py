import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SSD Plus")
    parser.add_argument("--num_classes", type=int, help="Number of classes", default=3)
    parser.add_argument("--lr", type=float, help="Learning rate", default=5e-4)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=8)
    parser.add_argument("--epoch", type=int, help="Epoch", default=100)
    parser.add_argument("--neg_radio", type=int, help="Negative radio", default=3)
    parser.add_argument("--save_folder", type=str, help="Save folder path", default="./weights/")
    parser.add_argument("--log_frequency", type=int, help="Log frequency", default=10)
    parser.add_argument("--min_size", type=int, help="Min size", default=300)
    parser.add_argument("--grids", type=tuple, help="Grids", default=(38, 19, 10, 5, 3, 1))
    parser.add_argument("--anchor_num", type=list, help="Anchor num", default=[4, 6, 6, 6, 4, 4])
    parser.add_argument("--mean", type=tuple, help="Mean", default=(104, 117, 123))
    parser.add_argument("--aspect_ratios", type=tuple, help="Aspect ratios", default=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)))
    parser.add_argument("--steps", type=list, help="Steps", default=[s / 300 for s in (8, 16, 32, 64, 100, 300)])
    parser.add_argument("--sizes", type=list, help="Sizes", default=[s / 300 for s in (30, 60, 111, 162, 213, 264, 315)])
    parser.add_argument("--variance", type=tuple, help="Variance", default=(0.1, 0.2))
    args = parser.parse_args()
    return args

args = parse_args()