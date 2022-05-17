from re import I
import time, os
import torch
import torch.optim as optim
import torch.utils.data

from util.tools import *
from train.loss import *
from train.scheduler import *
from dataloader.yolodata import *
from dataloader.data_transforms import *

from terminaltables import AsciiTable

class Trainer:
    def __init__(self, model, train_loader, eval_loader, hparam, class_str, device, checkpoint = None, torch_writer = None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam['max_batch']
        self.decay_step = hparam['steps']
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.torch_writer = torch_writer
        self.yololoss = YoloLoss(self.device, self.model.n_classes, hparam['ignore_cls'])
        # TODO 6 : Change Optimizer
        #self.optimizer = optim.SGD(model.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['decay'])
        self.optimizer = optim.Adam(model.parameters(), lr=hparam['lr'], weight_decay = hparam['decay'])
        #self.optimizer = optim.NAdam(model.parameters(), lr=hparam['lr'], weight_decay = hparam['decay'])
        self.class_str = class_str
        
        print("num_train_image:",len(train_loader)*hparam['batch'])
        
        
        
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iter = checkpoint['iteration']

        scheduler_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                           milestones=[10000,20000,30000],
                                                           gamma=0.5)
                                                            # gamma는 1/2씩 줄이겠다고 할당
                                                            
        #scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_batch-hparam['burn_in'])
        #TODO 4 : select scheduler
        #warm-up learning rate scheduler
        self.lr_scheduler = LearningRateWarmUP(optimizer=self.optimizer,
                                               warmup_iteration=hparam['burn_in'],
                                               target_lr=hparam['lr'],
                                               after_scheduler=scheduler_multistep)
                                               # 처음엔 아주 작은 값부터 시작해 원하는 학습률로 도달

    def run(self):
        while True:
            self.model.train()
            loss = self.run_iter()
            self.epoch += 1
            if self.epoch % 30 == 0:
                checkpoint_path = os.path.join("./output", "model_epoch" + str(self.epoch) + ".pth")
                torch.save({'epoch': self.epoch,
                            'iteration': self.iter,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss}, checkpoint_path)
                
                #evaluate
                self.model.eval()
                self.run_eval()
            # if iteration is greater than max_iteration, break
            if self.max_batch <= self.iter:
                break
    
    def run_iter(self):
        #torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(self.train_loader):
            #drop the invalid frames
            if batch is None:
                continue
            input_img, targets, anno_path = batch
            # TODO 10
            #print("shape:",input_img.shape)
            #show_img(input_img.to("cpu").numpy(),i)

            #sys.exit(1)
            # show the input image and bounding boxes on it

            # input_wh = [input_img.shape[3], input_img.shape[2]]
            # for b in range(input_img.shape[0]):
            #     target_box = targets[targets[:,0] == b,2:6]
            #     target_box[:,0] *= input_wh[0]
            #     target_box[:,2] *= input_wh[0]
            #     target_box[:,1] *= input_wh[1]
            #     target_box[:,3] *= input_wh[1]
            #     drawBox(input_img.detach().numpy()[b], target_box, cls = targets[targets[:,0] == b,1])
            # continue
            
            input_img = input_img.to(self.device, non_blocking=True)

            start_time = time.time()

            #inference model
            output = self.model(input_img)
            #compute loss
            loss, loss_list = self.yololoss.compute_loss(pred = output,
                                                        targets = targets,
                                                        yolo_layers = self.model.yolo_layers,
                                                        tmp_img = None)
            
            calc_time = time.time() - start_time

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step(self.iter)
            self.iter += 1

            if i % 100 == 0:
                duration = float(time.time() - start_time)
                latency = self.model.batch / duration
                print("loss : ", loss.item())
                print("epoch {} / iter {} lr {:.5f} , loss {:.5f} latency {:.5f}".format(self.epoch, self.iter, get_lr(self.optimizer), loss.item(), calc_time))
                self.torch_writer.add_scalar("lr", get_lr(self.optimizer), self.iter)
                self.torch_writer.add_scalar('example/sec', latency, self.iter)
                self.torch_writer.add_scalar('total_loss', loss, self.iter)
                loss_name = ['total_loss','obj_loss', 'cls_loss', 'box_loss']
                for ln, ls in zip(loss_name, loss_list):
                    self.torch_writer.add_scalar(ln, ls, self.iter)
        return loss
    
    def run_eval(self):
        #all predictions on eval dataset
        predict_all = []
        #all ground truth on eval dataset
        gt_labels = []
        for i, batch in enumerate(self.eval_loader):
            #skip invalid frames
            if batch is None:
                continue
            input_img, targets, _ = batch
            
            input_img = input_img.to(self.device, non_blocking=True)
            
            gt_labels += targets[...,1].tolist()

            targets[...,2:6] = cxcy2minmax(targets[...,2:6])
            input_wh = [input_img.shape[3], input_img.shape[2]]
            targets[...,2] *= input_wh[0]
            targets[...,4] *= input_wh[0]
            targets[...,3] *= input_wh[1]
            targets[...,5] *= input_wh[1]

            with torch.no_grad():
                output = self.model(input_img)
                best_box_list = non_max_suppression(output, conf_thres=0.1, iou_thres=0.5)
                
            predict_all += get_batch_statistics(best_box_list, targets, iou_threshold=0.5)
                
            if len(predict_all) == 0:
                print("no detection in eval data")
                return None
            if i % 100 == 0:
                print("-------eval {}th iter -----".format(i))
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*predict_all))]

        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, gt_labels)
        
        #print evaluation scores
        if metrics_output is not None:
            precision, recall, ap, f1, ap_class = metrics_output
            ap_table = [["Index", "Class", "AP", "Precision", "Recall", "f1"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, self.class_str[c], "%.5f" % ap[i], "%.5f" % precision[i], "%.5f" % recall[i], "%.5f" % f1[i]]]
            print(AsciiTable(ap_table).table)
        print("---- mAP {AP.mean():.5f} ----")

        for c, a, p, r, f in zip(self.class_str, ap, precision, recall, f1):
            self.torch_writer.add_scalars("Evaluation/AP", {c : a}, self.iter)
            self.torch_writer.add_scalars("Evaluation/Precision", {c : p}, self.iter)
            self.torch_writer.add_scalars("Evaluation/Recall", {c : r}, self.iter)
            self.torch_writer.add_scalars("Evaluation/F1", {c : f}, self.iter)
