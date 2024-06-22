import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class EqualizedFocalLoss(nn.Module):
    def __init__(self,
                 name='equalized_focal_loss',
                 loss_weight=1.0,
                 ignore_index=-1,
                 num_classes=80,
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 scale_factor=8.0,
                 fpn_levels=5):
        super().__init__()
        
        # cfg for focal loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # ignore bg class and ignore idx
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # cfg for efl loss
        self.scale_factor = scale_factor
        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes))

        # grad collect
        self.grad_buffer = []
        self.fpn_levels = fpn_levels
       

    def forward(self, input, target, normalizer=None):
        self.n_c = input.shape[-1]
        self.input = input.reshape(-1, self.n_c) # 마지막의 클래스의 총 수로 맞춰서 평탄화 작업
        
        # target의 크기 조정
        if target.dim() == 3 and target.size(-1) == self.n_c:
            self.target = target.argmax(dim=-1).reshape(-1)
        else:
            self.target = target.reshape(-1)

        self.n_i, _ = self.input.size()

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(pred.size(0), pred.size(1) + 1)
            gt_classes = gt_classes.long()
            target[torch.arange(pred.size(0)), gt_classes] = 1
            return target[:, 1:]

        expand_target = expand_label(self.input, self.target) # one-hot encoding이 된 target
        sample_mask = (self.target != self.ignore_index)

        inputs = self.input[sample_mask]
        targets = expand_target[sample_mask]
        self.cache_mask = sample_mask
        self.cache_target = expand_target

        pred = torch.sigmoid(inputs)
        pred_t = pred * targets + (1 - pred) * (1 - targets) # targets(one-hot encoding)이므로 실제 클래스에 대해서 예측 확률을 취하고, 나머지 클래스에 대해서 예측확률을 취함

        map_val = 1 - self.pos_neg.detach()
        dy_gamma = self.focal_gamma + self.scale_factor * map_val # r_b + s (1- g^j)
        dy_gamma = dy_gamma.to(inputs.device)  # dy_gamma를 inputs와 동일한 디바이스로 이동 
        # focusing factor
        ff = dy_gamma.view(1, -1).expand(self.n_i, self.n_c)[sample_mask]
        # weighting factor
        wf = ff / self.focal_gamma # r_b + s(1-g^j)/r_b

        # ce_loss
        # 로그 함수를 사용할 때 클램핑
        ce_loss = -torch.log(torch.clamp(pred_t, min=1e-10))
        cls_loss = ce_loss * torch.pow((1 - pred_t), ff.detach()) * wf.detach() # EFL(p_t) = -a*(1-p_t)^(r^j)*log(p_t)

        # to avoid an OOM error
        # torch.cuda.empty_cache()

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            cls_loss = alpha_t * cls_loss

        return cls_loss.mean(1).sum()

    def collect_grad(self, grad_in):
        bs = grad_in.shape[0]
        self.grad_buffer.append(grad_in.detach().permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes))
        if len(self.grad_buffer) == self.fpn_levels:
            target = self.cache_target[self.cache_mask]
            grad = torch.cat(self.grad_buffer[::-1], dim=1).reshape(-1, self.num_classes)

            grad = torch.abs(grad)[self.cache_mask]
            pos_grad = torch.sum(grad * target, dim=0)
            neg_grad = torch.sum(grad * (1 - target), dim=0)

            dist.all_reduce(pos_grad) # 분산 학습에서 각 프로세스가 계산한 그라디언트를 합산하여 모델의 파라미터 업데이트에 사용
            dist.all_reduce(neg_grad)

            self.pos_grad += pos_grad
            self.neg_grad += neg_grad
            self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)

            self.grad_buffer = []