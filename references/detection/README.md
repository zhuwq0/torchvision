
## Training
```
python train.py  --data-path=/home/zhuwq/Disk1T/BillDietrich/ --dataset coco_rock --model maskrcnn_resnet50_fpn --epochs 26    --lr-steps 16 22 --aspect-ratio-group-factor 3 --batch-size 1 --lr 0.01
```

## Evaluation
```
python predict.py  --data-path=/home/zhuwq/Disk1T/BillDietrich/ --dataset coco_rock --model maskrcnn_resnet50_fpn --resume output/checkpoint.pth  --batch-size 1
```
