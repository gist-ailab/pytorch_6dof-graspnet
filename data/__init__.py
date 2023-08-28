import torch.utils.data
from data.base_dataset import collate_fn
import threading


def CreateDataset(opt, is_train=True):
    """loads dataset class"""

    if opt.arch == 'vae' or opt.arch == 'gan':
        from data.grasp_sampling_data import GraspSamplingData, BimanualGraspSamplingData, \
        BimanualGraspSamplingDataV2, BimanualGraspSamplingDataV3, BimanualBlockGraspSamplingData,\
        BimanualAnchorGraspSamplingData, BimanualSecondGraspSamplingData
        if opt.is_bimanual:
            if opt.use_block:
                dataset = BimanualBlockGraspSamplingData(opt, is_train=is_train)
            elif opt.use_anchor:
                dataset = BimanualAnchorGraspSamplingData(opt, is_train=is_train)
            else:
                # select first or second grasp sampling
                if opt.second_grasp_sample:
                    dataset = BimanualSecondGraspSamplingData(opt, is_train=is_train)
                else:
                    dataset = BimanualGraspSamplingData(opt, is_train=is_train)
        elif opt.is_bimanual_v2 and not opt.is_bimanual_v3:
            dataset = BimanualGraspSamplingDataV2(opt, is_train=is_train)
        elif opt.is_bimanual_v3:
            dataset = BimanualGraspSamplingDataV3(opt, is_train=is_train) 
        else:
            dataset = GraspSamplingData(opt)
    else:
        from data.grasp_evaluator_data import GraspEvaluatorData, BimanualGraspEvaluatorDataset
        if opt.is_bimanual:
            dataset = BimanualGraspEvaluatorDataset(opt)
        else:
            dataset = GraspEvaluatorData(opt)
    return dataset


class DataLoader:
    """multi-threaded data loading"""
    def __init__(self, opt, is_train=True):
        self.opt = opt
        self.dataset = CreateDataset(opt, is_train)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.num_objects_per_batch,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                print('reached max dataset size')
                break
            yield data
