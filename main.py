# 1st, import modules based on ./demo/cub_s3n.yml
from trainer import network_trainer
from datasets import fetch_data, image_transform
from fgvc_datasets import fgvc_dataset  # Fine-grained visual classification dataset
from sss_net import s3n, three_stage
from losses import multi_smooth_loss
from optimizers import sgd_optimizer
from utils import finetune
from utility import multi_topk_meter
from meters import loss_meter
from hooks import update_lr, print_state, checkpoint
import inspect
import os
from pprint import pprint

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class WrapperFunc:
    # separate func param define and use
    def __init__(self, func, **kwargs):
        self.func = func
        # default func params, inverse order, as len(args) >= len(defaults)
        argspec = inspect.getfullargspec(func)
        args, default_vals = argspec.args, argspec.defaults
        vals = ['nerver_use'] * (len(args) - len(default_vals)) + list(default_vals)  # left add None for postition args
        params = {}
        for k, v in zip(args, vals):
            if k not in args:  # false pass in args
                continue
            params[k] = v if k not in kwargs else kwargs[k]  # update if pass in
        self.func_params = {
            k: v for k, v in params.items() if v != 'nerver_use'  # filter no_set args
        }
        pprint(self.func_params)

    def __call__(self, *args):
        return self.func(*args, **self.func_params)


def train():
    data_dir = './datasets/CUB_200_2011'

    train_transform = image_transform(
        image_size=[488, 488],
        augmentation={'horizontal_flip': 0.5,
                      'random_crop': {'scale': [0.5, 1]}},
    )
    test_transform = image_transform(
        image_size=[600, 600],
        augmentation={'center_crop': 448}
    )

    data_loaders = fetch_data(
        dataset=fgvc_dataset,  # pass in a class
        data_dir=data_dir,
        batch_size=2,
        num_workers=1,
        # batch_size=16,
        # num_workers=4,
        train_transform=train_transform,
        test_transform=test_transform,
        train_splits=['train'],
        test_splits=['test']
    )

    model = s3n(mode='resnet50',
                num_classes=200,
                task_input_size=448,
                radius=0.09, radius_inv=0.3)

    criterion = WrapperFunc(multi_smooth_loss, smooth_ratio=0.85)
    optimizer = WrapperFunc(sgd_optimizer, lr=0.01, momentum=0.9, weight_decay=1.e-4)
    parameter = WrapperFunc(finetune, base_lr=0.001, groups={'classifier': 10.0,
                                                             'radius': 0.0001,
                                                             'filter': 0.0001})
    meters = {
        'top1': WrapperFunc(multi_topk_meter, k=1, init_num=0),
        'loss': loss_meter
    }

    hooks = {  # Dict[str, List[Callable[[Context], None]]]  # these func pass in Context
        'on_start_epoch': [WrapperFunc(update_lr, epoch_step=40)],
        'on_start_forward': [three_stage],  # list of callable func
        'on_end_epoch': [
            WrapperFunc(print_state, formats={'epoch: {epoch_idx}',
                                              'train_loss: {metrics[train_loss]:.4f}',
                                              'test_loss: {metrics[test_loss]:.4f}',
                                              'train_branch1_top1: {metrics[train_top1][branch_0]:.2f}%',
                                              'train_branch2_top1: {metrics[train_top1][branch_1]:.2f}%',
                                              'train_branch3_top1: {metrics[train_top1][branch_2]:.2f}%',
                                              'train_branch4_top1: {metrics[train_top1][branch_3]:.2f}%',
                                              'test_branch1_top1: {metrics[test_top1][branch_0]:.2f}%',
                                              'test_branch2_top1: {metrics[test_top1][branch_1]:.2f}%',
                                              'test_branch3_top1: {metrics[test_top1][branch_2]:.2f}%',
                                              'test_branch4_top1: {metrics[test_top1][branch_3]:.2f}%', }),
            WrapperFunc(checkpoint, save_dir='./generate/s3n/', save_step=1)
        ]
    }

    network_trainer(
        data_loaders=data_loaders,
        log_path='./logs/cub_s3n.log',
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        parameter=parameter,
        meters=meters,
        max_epoch=60,
        device='cuda',
        use_data_parallel=True,
        hooks=hooks
    )


if __name__ == '__main__':
    train()
