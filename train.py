import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import parse_args
from model import SSDPlus, MultiBoxLoss
from utils import detection_collate, detect, preproc_for_test, MultiBoxEncoder, adjust_learning_rate1, adjust_learning_rate2
from voc_eval import voc_eval
from voc_dataset import VOCDetection, VOC_LABELS



device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # ---------------------------- train --------------------------------
    
    model = SSDPlus(args.num_classes, args.anchor_num, pretrain=True).to(device)
    mbe = MultiBoxEncoder(args)
        
    dataset = VOCDetection(args, 'dataset', name='train', is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=detection_collate, num_workers=4)

    val_dataset = VOCDetection(args, 'dataset', name='val', is_train=False)

    criterion = MultiBoxLoss(args.num_classes, args.neg_radio).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -----------------------------------------------------------------

    # ---------------------------- val --------------------------------

    annopath = os.path.join('dataset', 'Annotations', "%s.xml")  
    imgpath = os.path.join('dataset', 'JPEGImages', '%s.jpg')    
    imgsetpath = os.path.join('dataset', 'ImageSets', 'Main', '{:s}.txt')   
    cachedir = os.path.join(os.getcwd(), 'annotations_cache')  

    output_dir = 'result'

    os.makedirs(output_dir, exist_ok=True)
    
    # -----------------------------------------------------------------

    print('start training........')
    for e in range(args.epoch):
        model.train()

        if e == 70:
            adjust_learning_rate1(optimizer, args.lr)
        elif e == 90:
            adjust_learning_rate2(optimizer, args.lr)

        total_loc_loss = 0
        total_cls_loss = 0
        total_loss = 0
        for i , (img, boxes) in enumerate(dataloader):
            img = img.to(device)
            gt_boxes = []
            gt_labels = []
            for box in boxes:
                labels = box[:, 4]
                box = box[:, :-1]
                match_loc, match_label = mbe.encode(box, labels)
            
                gt_boxes.append(match_loc)
                gt_labels.append(match_label)
            
            gt_boxes = torch.FloatTensor(gt_boxes).to(device)
            gt_labels = torch.LongTensor(gt_labels).to(device)


            p_loc, p_label = model(img)


            loc_loss, cls_loss = criterion(p_loc, p_label, gt_boxes, gt_labels)

            loss = loc_loss + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()
            if i % args.log_frequency == 0:
                avg_loc = total_loc_loss / (i+1)
                avg_cls = total_cls_loss / (i+1)
                avg_loss = total_loss / (i+1)
                print('epoch[{}] | batch_idx[{}] | loc_loss [{:.2f}] | cls_loss [{:.2f}] | total_loss [{:.2f}]'.format(e, i, avg_loc, avg_cls, avg_loss))

        if e % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_folder, '{}-loss-{:.2f}.pth'.format(e, total_loss)))

            files = [open(os.path.join(output_dir, '{:s}.txt'.format(label)), mode='w') for label in VOC_LABELS]
            model.eval()
            for i in tqdm(range(len(val_dataset))):
                src = val_dataset[i][0]
                
                img_name = os.path.basename(val_dataset.ids[i][0]).split('.')[0]
                image = preproc_for_test(src, args.min_size, args.mean)
                image = torch.from_numpy(image).to(device)
                with torch.no_grad():
                    loc, conf = model(image.unsqueeze(0))
                loc = loc[0]
                conf = conf[0]
                conf = F.softmax(conf, dim=1)
                conf = conf.cpu().numpy()
                loc = loc.cpu().numpy()

                decode_loc = mbe.decode(loc)
                gt_boxes, gt_confs, gt_labels = detect(decode_loc, conf, nms_threshold=0.5, gt_threshold=0.01)

                #no object detected
                if len(gt_boxes) == 0:
                    continue

                h, w = src.shape[:2]
                gt_boxes[:, 0] = gt_boxes[:, 0] * w
                gt_boxes[:, 1] = gt_boxes[:, 1] * h
                gt_boxes[:, 2] = gt_boxes[:, 2] * w
                gt_boxes[:, 3] = gt_boxes[:, 3] * h


                for box, label, score in zip(gt_boxes, gt_labels, gt_confs):
                    print(img_name, "{:.3f}".format(score), "{:.1f} {:.1f} {:.1f} {:.1f}".format(*box), file=files[label])


            for f in files:
                f.close()
            
            
            print('start cal MAP.........')
            aps = []
            for f in os.listdir(output_dir):
                filename = os.path.join(output_dir, f)
                class_name = f.split('.')[0]
                rec, prec, ap = voc_eval(filename, annopath, imgsetpath.format('test'), class_name, cachedir, ovthresh=0.1, use_07_metric=True)
                print(class_name, ap)
                aps.append(ap)

            print('mean MAP is : ', np.mean(aps))

    # nohup python train.py > train.log 2>&1 &



