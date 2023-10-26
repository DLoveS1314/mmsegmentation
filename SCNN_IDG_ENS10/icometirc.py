
'''
https://mmengine.readthedocs.io/zh_CN/latest/design/evaluation.html
评测指标有两个重要函数 需要继承


process() 用于处理每个批次的测试数据和模型预测结果。处理结果应存放在 self.results 列表中，用于在处理完所有测试数据后计算评测指标。该方法具有以下 2 个参数：

data_batch：一个批次的测试数据样本，通常直接来自与数据加载器

data_samples：对应的模型预测结果 该方法没有返回值。函数接口定义如下：
process 在 test_loop 中的run_iter调用 每一个batch处理一次


compute_metrics() 用于计算评测指标，并将所评测指标存放在一个字典中返回。该方法有以下 1 个参数：

results：列表类型，存放了所有批次测试数据经过 process() 方法处理后得到的结果 该方法返回一个字典，里面保存了评测指标的名称和对应的评测值。函数接口定义如下：

compute_metrics 在 test_loop 中的 run 的 self.evaluator.evaluate 调用  

'''
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from .loss import CrpsGaussianLoss,EECRPSGaussianLoss

from mmseg.registry import METRICS

def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value

@METRICS.register_module()
class CRPS_metric(BaseMetric):
    r"""    Args:

        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
 
    """
    default_prefix: Optional[str] = 'crps'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.crps = CrpsGaussianLoss(mode='mean')
        self.eecrps = None 
        # EECRPSGaussianLoss()
        assert self.eecrps is  None, 'EECRPSGaussianLoss 尚未 完成'
    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples. 
        
        testloop中 run_iter调用 
        每批次计算一次结果，计算的结果放到self.results
        process() 用于处理每个批次的测试数据和模型预测结果。处理结果应存放在 self.results 列表中，
        用于在处理完所有测试数据后计算评测指标。该方法具有以下 2 个参数：
        self.results 中存放的具体类型取决于评测指标子类的实现。
        例如，当测试样本或模型输出数据量较大（如语义分割、图像生成等任务），
        不宜全部存放在内存中时，可以在 self.results 中存放每个批次计算得到的指标，
        并在 compute_metrics() 中汇总；或将每个批次的中间结果存储到临时文件中，
        并在 self.results 中存放临时文件路径，最后由 compute_metrics() 从文件中读取数据并计算指标。
        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_label = data_sample['pred_sem_seg']['data'] # 2 ,H,W  包含均值和方差
            gt_label = data_sample['gt_sem_seg']['data'].to(pred_label) # H W 只包含真值
            if gt_label.dim() == 3:
                gt_label = gt_label.squeeze(0)
            result['pred_label'] = pred_label
            result['gt_label'] = gt_label
            # Save the result to `self.results`.
            # result = self._cal(pred=pred_label,target=gt_label)
            self.results.append(result)
    def _stack_batch_gt(self, results) :
        target = torch.stack([res['gt_label']   for res in results ],dim=0) # H,W => N, H,W
        pred = torch.stack([res['pred_label']  for res in results ],dim=0) #2 ,H,W => N,2,H,W (后续要拆开成两个)
        return target,pred
    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.
            对每批次计算的的结果在进行计算 得到最终的结果
        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
         
        # torch.stack(gt_semantic_segs, dim=0)
        # target = torch.cat([res['gt_label'] for res in results])
        # pred = torch.cat([res['pred_label'] for res in results])
        target ,pred = self._stack_batch_gt(results)
        # print('compute_metricstarget',target.shape)
        # print('compute_metrics pred',pred.shape)
        metrics = self._cal(pred=pred,target=target)
        # metrics['ecrps'] = crps_value.item()
        return metrics
    def _cal(self,pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence]):
            pred = to_tensor(pred)
            target = to_tensor(target)
            num = pred.size(0)
            assert pred.size(0) == target.size(0), \
                f"The size of pred ({pred.size(0)}) doesn't match "\
                f'the target ({target.size(0)}).'
            results = {}
            crps_value = self.crps(pred_mean=pred[:,0,...],pred_stddev=pred[:,1,...],target=target).item()
            results.update({'crps': crps_value})
            # 后续可能要加上EECRPSGaussianLoss
            
            return results
            
    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the crps.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).

        Returns:
            torch.Tensor | List[List[torch.Tensor]]: crps.

            - torch.Tensor: If the ``pred`` is a sequence of label instead of
              score (number of dimensions is 1). Only return a top-1 accuracy
              tensor, and ignore the argument ``topk` and ``thrs``.
            - List[List[torch.Tensor]]: If the ``pred`` is a sequence of score
              (number of dimensions is 2). Return the accuracy on each ``topk``
              and ``thrs``. And the first dim is ``topk``, the second dim is
              ``thrs``.
        """
        assert NotImplementedError
        crps = CrpsGaussianLoss()
        pred = to_tensor(pred)
        target = to_tensor(target) 
        num = pred.size(0)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target.size(0)}).'
        results = {}
        crps_value = crps(pred_mean=pred[:,0,...],pred_stddev=pred[:,1,...],target=target)
        results.update({'crps': crps_value})
        # 后续可能要加上EECRPSGaussianLoss
        return results