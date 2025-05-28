### `try/batch.py`

Evaluate: Input a Batch is not equal to Input the Sum of a Batch

### `pytorch/dataset/dataset_dataloader2.py`

Implement a Dataset and a Module

### `pytorch/module/tutorial.py`

Implement a tutrial to build a model with module and parameter

### `distributed/speed.py`

A distributed training framework based on deepspeed

+ args required in `deepspeed`
```
error: unrecognized arguments: --local_rank=0 --deepspeed_config ..
```
+ specify default cuda device to avoid put of memory
```python
torch.cuda.set_device(args.local_rank)
```