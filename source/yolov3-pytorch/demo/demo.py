import torch
from train.loss import *

class Demo:
    def __init__(self, model, data, data_loader, device, hparam):
        self.model = model
        self.class_list = data.class_str
        self.data_loader = data_loader
        self.device = device
        self.yololoss = YoloLoss(self.device, self.model.n_classes, hparam['ignore_cls'])
        self.preds = None

    def run(self):
        for i, batch in enumerate(self.data_loader):
            #drop the invalid frames
            if batch is None:
                continue
            input_img, _, _ = batch
            
            #drawBox(input_img.detach().numpy()[0])
            #np.save("torch_input.npy",input_img.detach().numpy())
            
            input_img = input_img.to(self.device, non_blocking=True)

            num_batch = input_img.shape[0]
            with torch.no_grad():
                output = self.model(input_img)
                best_box_list = non_max_suppression(output,
                                                    conf_thres=0.4,
                                                    iou_thres=0.45)

                for b in range(num_batch):
                    if best_box_list[b] is None:
                        continue
                    print(best_box_list[b])
                    final_box_list = [bbox for bbox in best_box_list[b] if bbox[4] > 0.5]
                    print("final :", final_box_list)

                    if final_box_list is None:
                        continue
                    show_img = input_img[b].detach().cpu().numpy()
                    drawBoxlist(show_img, final_box_list, mode=1, name = str(i)+"_"+str(b))
