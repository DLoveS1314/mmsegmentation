
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
    r"""Accuracy evaluation metric.

    For either binary classification or multi-class classification, the
    accuracy is the fraction of correct predictions in all predictions:

    .. math::

        \text{Accuracy} = \frac{N_{\text{correct}}}{N_{\text{all}}}

    Args:
        topk (int | Sequence[int]): If the ground truth label matches one of
            the best **k** predictions, the sample will be regard as a positive
            prediction. If the parameter is a tuple, all of top-k accuracy will
            be calculated and outputted together. Defaults to 1.
        thrs (Sequence[float | None] | float | None): If a float, predictions
            with score lower than the threshold will be regard as the negative
            prediction. If None, not apply threshold. If the parameter is a
            tuple, accuracy based on all thresholds will be calculated and
            outputted together. Defaults to 0.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        >>> import torch
        >>> from mmcls.evaluation import Accuracy
        >>> # -------------------- The Basic Usage --------------------
        >>> y_pred = [0, 2, 1, 3]
        >>> y_true = [0, 1, 2, 3]
        >>> Accuracy.calculate(y_pred, y_true)
        tensor([50.])
        >>> # Calculate the top1 and top5 accuracy.
        >>> y_score = torch.rand((1000, 10))
        >>> y_true = torch.zeros((1000, ))
        >>> Accuracy.calculate(y_score, y_true, topk=(1, 5))
        [[tensor([9.9000])], [tensor([51.5000])]]
        >>>
        >>> # ------------------- Use with Evalutor -------------------
        >>> from mmcls.structures import ClsDataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     ClsDataSample().set_gt_label(0).set_pred_score(torch.rand(10))
        ...     for i in range(1000)
        ... ]
        >>> evaluator = Evaluator(metrics=Accuracy(topk=(1, 5)))
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(1000)
        {
            'accuracy/top1': 9.300000190734863,
            'accuracy/top5': 51.20000076293945
        }
    """
    default_prefix: Optional[str] = 'crps'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

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
            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']
            if 'score' in pred_label:
                result['pred_score'] = pred_label['score'].cpu()
            else:
                result['pred_label'] = pred_label['label'].cpu()
            result['gt_label'] = gt_label['label'].cpu()
            # Save the result to `self.results`.
            self.results.append(result)

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
        metrics = {}

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])

            try:
                acc = self.calculate(pred, target, self.topk, self.thrs)
            except ValueError as e:
                # If the topk is invalid.
                raise ValueError(
                    str(e) + ' Please check the `val_evaluator` and '
                    '`test_evaluator` fields in your config file.')

            multi_thrs = len(self.thrs) > 1
            for i, k in enumerate(self.topk):
                for j, thr in enumerate(self.thrs):
                    name = f'top{k}'
                    if multi_thrs:
                        name += '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                    metrics[name] = acc[i][j].item()
        else:
            # If only label in the `pred_label`.
            pred = torch.cat([res['pred_label'] for res in results])
            acc = self.calculate(pred, target, self.topk, self.thrs)
            metrics['top1'] = acc.item()

        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        topk: Sequence[int] = (1, ),
        thrs: Sequence[Union[float, None]] = (0., ),
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the accuracy.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            thrs (Sequence[float | None]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. None means no thresholds.
                Defaults to (0., ).
            thrs (Sequence[float]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. Defaults to (0., ).

        Returns:
            torch.Tensor | List[List[torch.Tensor]]: Accuracy.

            - torch.Tensor: If the ``pred`` is a sequence of label instead of
              score (number of dimensions is 1). Only return a top-1 accuracy
              tensor, and ignore the argument ``topk` and ``thrs``.
            - List[List[torch.Tensor]]: If the ``pred`` is a sequence of score
              (number of dimensions is 2). Return the accuracy on each ``topk``
              and ``thrs``. And the first dim is ``topk``, the second dim is
              ``thrs``.
        """

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        num = pred.size(0)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            # For pred label, ignore topk and acc
            pred_label = pred.int()
            correct = pred.eq(target).float().sum(0, keepdim=True)
            acc = correct.mul_(100. / num)
            return acc
        else:
            # For pred score, calculate on all topk and thresholds.
            pred = pred.float()
            maxk = max(topk)

            if maxk > pred.size(1):
                raise ValueError(
                    f'Top-{maxk} accuracy is unavailable since the number of '
                    f'categories is {pred.size(1)}.')

            pred_score, pred_label = pred.topk(maxk, dim=1)
            pred_label = pred_label.t()
            correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
            results = []
            for k in topk:
                results.append([])
                for thr in thrs:
                    # Only prediction values larger than thr are counted
                    # as correct
                    _correct = correct
                    if thr is not None:
                        _correct = _correct & (pred_score.t() > thr)
                    correct_k = _correct[:k].reshape(-1).float().sum(
                        0, keepdim=True)
                    acc = correct_k.mul_(100. / num)
                    results[-1].append(acc)
            return results
