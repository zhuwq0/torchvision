r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import presets
import torch
import torch.utils.data
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco, get_coco_api_from_dataset, get_coco_kp
from transforms import SimpleCopyPaste

import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs, fname='test.png'):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.savefig(fname, dpi=300)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(images)

        proba_threshold = 0.01
        score_threshold = .8

        for image, output in zip(images, outputs):
            bool_masks = output['masks'] > proba_threshold
            print(f"shape = {bool_masks.shape}, dtype = {bool_masks.dtype}")

            # There's an extra dimension (1) to the masks. We need to remove it
            bool_masks = bool_masks.squeeze(1)

            # convert image from float32 to unit8
            image = (image - image.min())/(image.max() - image.min()) * 255.0
            image = image.to(torch.uint8)
            show(draw_segmentation_masks(image.to("cpu"), bool_masks, alpha=0.9), "test_mask.png")
            show(draw_bounding_boxes(image.to("cpu"), boxes=output['boxes'][output['scores'] > score_threshold], width=4), "test_bbox.png")

            raise
        raise

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        print(coco_evaluator.eval_imgs)
        raise

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    torch.set_num_threads(n_threads)
    return 0


def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


def get_dataset(name, image_set, transform, data_path):
    paths = {"coco": (data_path, get_coco, 91), "coco_kp": (data_path, get_coco_kp, 2), "coco_rock": (data_path, get_coco, 2), }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="coco", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./results", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="./output/checkpoint.pth", type=str, help="path of checkpoint")
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    return parser


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    dataset_test, num_classes = get_dataset(args.dataset, "val", get_transform(False, args), args.data_path)

    print("Creating data loaders")
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Creating model")
    kwargs = {}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    model = torchvision.models.get_model(
        args.model, weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes, **kwargs
    )
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    checkpoint = torch.load(args.resume, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"])


    torch.backends.cudnn.deterministic = True
    evaluate(model, data_loader_test, device=device)

    return


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
