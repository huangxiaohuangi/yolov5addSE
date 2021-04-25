import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
    compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
    check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution)
from utils.google_utils import attempt_download
from utils.torch_utils import init_seeds, ModelEMA, select_device, intersect_dicts
import warnings

warnings.filterwarnings('ignore')


def train(hyp, opt, device, tb_writer=None):
    print(f'Hyperparameters {hyp}')
    # 获取记录训练日志的路径
    """
        训练日志包括：权重、tensorboard文件、超参数hyp、设置的训练参数opt(也就是epochs,batch_size等),result.txt
        result.txt包括: 占GPU内存、训练集的GIOU loss, objectness loss, classification loss, 总loss, 
        targets的数量, 输入图片分辨率, 准确率TP/(TP+FP),召回率TP/P ; 
        测试集的mAP50, mAP@0.5:0.95, GIOU loss, objectness loss, classification loss.
        还会保存batch<3的ground truth
    """
    # 如果设置进化算法则不会传入tb_writer（则为None），设置一个evolve文件夹作为日志目录
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # 训练日志文件
    # 设置保存权重的路径
    wdir = str(log_dir / 'weights') + os.sep
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    # 设置保存results的路径
    results_file = str(log_dir / 'results.txt')
    # 获取轮次、批次、总批次（涉及到分布式训练）、权重、进程号（主要用于分布式训练）
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # TODO: Use DDP logging. Only the first process is allowed to log.
    # 保存hyp和opt
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # 配置
    cuda = device.type != 'cpu'
    # 设置随机种子
    init_seeds(2 + rank)
    # 加载数据配置信息
    with open(opt.data, encoding='UTF-8') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    # torch_distributed_zero_first 同步所有进程
    # check_dataset检查数据集，如果没找到数据集则下载数据集（仅适用于项目中自带的yaml文件数据集）
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)
    # 获取训练集、测试集图片路径
    train_path = data_dict['train']
    test_path = data_dict['val']
    # 获取类别数量和类别名字
    # 如果设置了opt,single_cls则为一类
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # 模型
    pretrained = weights.endswith('.pt')
    # 如果采用预训练
    if pretrained:
        # 加载模型，从google云盘中自动下载模型
        # 但通常会下载失败,建议提前下载下来放进weights目录
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # 本地不存在就选择下载
        # 数据加载检查点
        ckpt = torch.load(weights, map_location=device)  # 模型加载
        """
        这里模型创建，可通过opt.cfg，也可通过ckpt['model'].yaml
        这里的区别在于是否是resume，resume时会将opt.cfg设为空，
        则按照ckpt['model'].yaml创建模型；
        这也影响着下面是否除去anchor的key(也就是不加载anchor)，如果resume则不加载anchor
        主要是因为保存的模型会保存anchors，有时候用户自定义了anchor之后，再resume，则原来基于coco数据集的anchor就会覆盖自己设定的anchor，
        参考https://github.com/ultralytics/yolov5/issues/459
        所以下面设置了intersect_dicts，该函数就是忽略掉exclude
        """
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)
        exclude = ['anchor'] if opt.cfg else []
        state_dict = ckpt['model'].float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)
        # 显示加载预训练权重的键值对和创建模型的键值对
        # 如果设置了resume,则会少家贼两个键值对(anchors,anchor_grid)
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))
    else:
        # 创建模型，ch为输入图片通道
        model = Model(opt.cfg, ch=3, nc=nc).to(device)

    """
    nbs为模拟的batch_size
    就比如默认的话上面设置的opt.batch_size为16，这个nbs就位64
    也就是模型梯度累计了64/16=4（accumulate）次之后
    再更新一次模型，变相的扩大了batch_size
    """
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)
    # 根据accumulate设置权重衰减系数
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs

    pg0, pg1, pg2 = [], [], []  # 优化器参数
    # 将模型分成三组（weights、bn、bias，其它所有参数）优化
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)
        else:
            pg0.append(v)
    # 选择使用优化器，并设置pg0组的优化方式
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # 设置weight、bn的优化方式
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    #  设置biases的优化方式
    optimizer.add_param_group({'params': pg2})
    # 打印优化信息
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2
    # 设置学习率衰减
    print('epochs ', epochs)
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 初始化开始训练的epoch和最好的结果
    # best_fitness是以[0.0, 0.0, 0.1, 0.9]为系数并乘以[精确度，召回率，mAP@0.5, mAP@0.5:0.95]再求和所得
    # 根据best_fitness来保存best.pt
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # 加载优化器与best_fitness
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        # 加载训练结果result.txt
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # 将结果写入文件
        # 加载训练的轮次
        start_epoch = ckpt['epoch'] + 1
        """
        如果resume，则备份权重          
        """
        """
           如果新设置epochs小于加载的epoch，
           则视新设置的epochs为需要再训练的轮次数而不再是总的轮次数
        """
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']

        del ckpt, state_dict

    # 获取模型总步长和模型输入图片分辨率
    gs = int(max(model.stride))  # 网格大小
    # 检查输入图片分辨率确保能够整除步长gs
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]
    print("imgse", imgsz)
    # DP 模式
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 使用跨卡同步BN
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print('Using SyncBatchNorm()')
    # 为模型创建EMA指数滑动平均，如果进程数大于1，则不创建
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP 模式
    # 如果rank不等于-1,则使用DistributedDataParallel模式
    # local_rank为gpu编号,rank为进程,例如rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU。
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=(opt.local_rank))

    # 创建加载器
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
                                            cache=opt.cache_images, rect=opt.rect, local_rank=rank,
                                            world_size=opt.world_size)
    # print('num_workers', dataloader.num_workers)
    """
    获取标签中最大的类别值，并于类别数作比较
    如果小于类别数则表示有问题
    """
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # 最大标记类别
    nb = len(dataloader)  # 批处理数

    # 测试加载器
    if rank in [-1, 0]:
        # local_rank设置为-1。因为只希望对第一个过程进行评估
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt, hyp=hyp, augment=False,
                                       cache=opt.cache_images, rect=True, local_rank=-1, world_size=opt.world_size)[0]

    # 模型参数
    hyp['cls'] *= nc / 80.
    # 设置类别数、超参数
    model.nc = nc
    model.hyp = hyp  # 将超参数附加到模型
    """
       设置giou的值在objectness loss中做标签的系数, 使用代码如下
       tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)
       这里model.gr=1，也就是说完全使用标签框与预测框的giou值来作为该预测框的objectness标签
       """
    model.gr = 1.0
    # 根据labels初始化图片采样权重
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    # 获取类别名称
    model.names = names

    if rank in [-1, 0]:
        # 将所有样本的标签拼接到一起shape为（total，5），统计后做可视化
        labels = np.concatenate(dataset.labels, 0)
        # 获得所有样本的类别
        c = torch.tensor(labels[:, 0])
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        # 根据上面的统计对所有样本的类别，中心点xy位置，长宽wh做可视化
        plot_labels(labels, save_dir=log_dir)
        if tb_writer:
            # tb_writer.add_hparams(hyp, {})
            tb_writer.add_histogram('classes', c, 0)
        """
        计算默认锚点anchor与数据集标签框的长宽比值
        标签的长h宽w与anchor的长h_a宽w_a的比值, 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
        如果标签框满足上面条件的数量小于总数的99%，则根据k-mean算法聚类新的锚点anchor
        """
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # 开始训练
    t0 = time.time()
    # 获取训练的迭代次数
    nw = max(3 * nb, 1e3)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    # 初始化mAP和results
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    '''
    设置学习率衰减所进行到的轮次
    目的是打断训练后--resume接着训练也能正常的衔接之前的训练进行学习率衰减
    
    '''
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    """
    打印训练和测试输入的图片分辨率
    加载图片时调用的cpu进程数
    从哪个epoch开始训练    
    """
    if rank in [0, -1]:
        print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
        print('Using %g dataloader workers' % dataloader.num_workers)
        print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # 更新图像权重
        if dataset.image_weights:
            # 产生索引
            """
            如果设置进行图片采样策略
            则根据前面初始化的图片采样权重model.class_weights以及maps配合每张图片包含的类别数
            通过random.choices生成图片索引indices从而进行采样            
            """
            if rank in [-1, 0]:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # 类别权重
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                                 k=dataset.n)
            # 如果是DDp就进行广播
            if rank != -1:
                indices = torch.zeros([dataset.n], dtype=torch.int)
                if rank == 0:
                    indices[:] = torch.from_tensor
                # 广播索引到其它group
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
        # 初始化训练师打印的平均损失信息
        mloss = torch.zeros(4, device=device)  # 平均损失
        if rank != -1:
            # DDP模式下打乱数据, ddp.sampler的随机采样数据是基于epoch+seed作为随机种子，
            # 每次epoch不同，随机种子就不同
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            # tqdm 创建进度条，方便训练时信息的展示
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=nb)  # 进度条
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # 计算迭代的次数iteration
            ni = i + nb * epoch  # 批处理号
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            """
            热身训练(前nw次迭代)
            在前nw次迭代中，根据以下方式选取accumulate和学习率
            """
            if ni <= nw:
                xi = [0, nw]
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    """
                    bias的学习率从0.1下降到基准学习率lr*lf(epoch)，
                    其他的参数学习率从0增加到lr*lf(epoch).
                    lf为上面设置的余弦退火的衰减函数
                    """
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # 设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # 比例因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # 混合精度
            with amp.autocast(enabled=cuda):
                # 前向传播
                pred = model(imgs)

                # 损失
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                if rank != -1:
                    # 平均不同gpu之间的梯度
                    loss *= opt.world_size  # DDP模式下设备之间的平均梯度
                # if not torch.isfinite(loss):
                #     print('WARNING: non-finite loss, ending training ', loss_items)
                #     return results

            # 反向传播
            scaler.scale(loss).backward()

            # 模型反向传播accumulate次之后再根据累积的梯度更新一次参数
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均损失
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                # 进度条显示以上信息
                pbar.set_description(s)

                # Plot
                # 将前三次迭代batch的标签框在图片上画出来并保存
                if ni < 3:
                    f = str(log_dir / ('train_batch%g.jpg' % ni))  # 文件名
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------
        # 进行学习率衰减
        scheduler.step()

        # DDP进程0或单GPU
        if rank in [-1, 0]:

            if ema is not None:
                # 更新EMA的属性，添加include的属性
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            # 判断该epoch是否为最后一轮
            final_epoch = epoch + 1 == epochs
            # 对测试集进行测试，计算mAP等指标
            # 测试时使用的是EMA模型
            if not opt.notest or final_epoch:  # 计算 mAP
                results, maps, times = test.test(opt.data,
                                                 batch_size=total_batch_size,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=log_dir)

            # 将指标写入result.txt
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            # 如果设置opt.bucket,上传results.txt到谷歌云盘
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # 添加指标，损失等信息到tensorboard显示
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # 更新best_fitness
            fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi

            # 模型保存
            """
            保存模型，还保存了epoch，results，optimizer等信息，
            optimizer将不会在最后一轮完成后保存
            model保存的是EMA的模型
            """
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:  # 创建检查点
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module if hasattr(ema, 'module') else ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                # 保存最后，最好并删除
                torch.save(ckpt, last, _use_new_zipfile_serialization=False)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # 训练结束

    if rank in [-1, 0]:
        """
        模型训练完后，strip_optimizer函数将optimizer从ckpt中去除；
        并且对模型进行model.half(), 将Float32的模型->Float16，
        可以减少模型大小，提高inference速度
        """
        n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # 重命名
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # 更新
        # 完成
        # 可视化results.txt文件
        if not opt.evolve:
            plot_results(save_dir=log_dir)  # 保存 results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    # 释放显存
    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    """
    cfg:模型配置文件，网络结构
    data：数据集配置文件，数据集的路径，类名等
    hyp：超参数文件
    epochs：训练总轮次
    batch-size：批次大小
    img-size：输入图片的分辨率大小
    rect：是否采用矩形训练，默认为False
    resume：接着打断训练上次的结果接着训练
    nosave：不保存模型。默认为False
    notest：不进行test，默认为False
    noautoanchor：不自动调整anchor，默认为False
    evolve：是否进行超参数进化，默认为False
    bucket：谷歌云盘bucket，一般不会用到
    cache-images：是否提前缓存图片到内存，以加快训练速度，默认为False
    weights：加载的权重文件
    num_workers   name：数据集的名字，如果设置：results.txt to results_name.txt，默认无
    device：训练的设备。cpu，0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
    multi-scale:是否进行多尺度训练，默认False
    single-cls:数据集是否只有一个类别，默认False
    adam:是否使用adam优化器
    sync-bn:是否使用跨卡同步BN,在DDP模式使用
    local_rank:gpu编号
    logdir:存放日志的目录
    workers:dataloader的最大worker数量
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5l.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5l.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/last.pt, or most recent run if blank')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    opt = parser.parse_args()

    # 是否resume
    if opt.resume:
        # 如果resume是str,则表示传入的是模型的路径地址
        # get_latest_run()函数获取runs文件夹中最近的last.pt
        last = get_latest_run() if opt.resume == 'get_last' else opt.resume  # 从最近一次运行中恢复
        if last and not opt.weights:
            print(f'Resuming training from {last}')
        opt.weights = last if opt.resume and not opt.weights else opt.weights
    if opt.local_rank == -1 or ("RANK" in os.environ and os.environ["RANK"] == "0"):
        check_git_status()
    # 获取超参数列表
    opt.hyp = opt.hyp or ('data/hyp.finetune.yaml' if opt.weights else 'data/hyp.scratch.yaml')
    # 检查配置文件信息
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # 检查文件
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    # 扩展image_size为[image_size, image_size]一个是训练size，一个是测试size
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
    device = select_device(opt.device, batch_size=opt.batch_size)

    opt.total_batch_size = opt.batch_size
    # 设置num_works的数量
    opt.world_size = 8
    opt.global_rank = -1

    # DDP 模式
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        # 根据gpu编号选择设备
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        # 初始化进程组
        dist.init_process_group(backend='nccl', init_method='env://')  # 分布式后端
        # 将总批次按照进程数分配给各个gpu
        opt.world_size = dist.get_world_size()
        opt.global_rank = dist.get_rank()
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size
    # 打印opt参数信息
    print(opt)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # 加载超参数

    # 训练
    # 如果不进行超参数进化，则直接调用train（）函数，开始训练
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            # 创建tensorboard
            print('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt.logdir)
            tb_writer = SummaryWriter(log_dir=increment_dir(Path(opt.logdir) / 'exp', opt.name))

        train(hyp, opt, device, tb_writer)

    # 更新超参数
    else:
        # 超参数进化列表，括号分别为（突变规模，最小值，最大值）
        meta = {'lr0': (1, 1e-5, 1e-1),  # (SGD=1E-2, Adam=1E-3) 学习率
                'momentum': (0.1, 0.6, 0.98),  # SGD momentum/Adam beta1 学习率动量
                'weight_decay': (1, 0.0, 0.001),  # 优化权重衰减器
                'giou': (1, 0.02, 0.2),  # GIoU损失系数
                'cls': (1, 0.2, 4.0),  # 分类损失系数
                'cls_pw': (1, 0.5, 2.0),  # 分类BCELoss中正样本的权重
                'obj': (1, 0.2, 4.0),  # 有无物体损失的系数
                'obj_pw': (1, 0.5, 2.0),  # 有无物体BCELoss中正样本的权重
                'iou_t': (0, 0.1, 0.7),  # 标签与anchors的iou阈值iou training threshold
                'anchor_t': (1, 2.0, 8.0),
                # 标签的长h宽w/anchor的长h_a宽w_a阈值, 即h/h_a, w/w_a都要在(1/2.26, 2.26)之间anchor-multiple threshold
                'fl_gamma': (0, 0.0, 2.0),  # 设为0则表示不使用focal loss(efficientDet default is gamma=1.5)
                # 下面是一些数据的增强的系数，包括颜色空间和图片空间
                'hsv_h': (1, 0.0, 0.1),  # 色调
                'hsv_s': (1, 0.0, 0.9),  # 饱和度
                'hsv_v': (1, 0.0, 0.9),  # 亮度
                'degrees': (1, 0.0, 45.0),  # 旋转角度
                'translate': (1, 0.0, 0.9),  # 水平和垂直平移
                'scale': (1, 0.0, 0.9),  # 缩放
                'shear': (1, 0.0, 10.0),  # 图像剪切
                'perspective': (1, 0.0, 0.001),  # 透视变换参数
                'flipud': (0, 0.0, 1.0),  # 图像上下翻转
                'fliplr': (1, 0.0, 1.0),  # 图像左右翻转
                'mixup': (1, 0.0, 1.0)}  # mixup系数

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # 仅测试/保存最终轮次
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # 演化指数
        yaml_file = Path('runs/evolve/hyp_evolved.yaml')  # 保存最好的结果
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # 如果存在则下载volve.txt
        # 默认进化100次
        """
         这里的进化算法是：根据之前训练时的hyp来确定一个base hyp再进行突变；
         如何根据？通过之前每次进化得到的results来确定之前每个hyp的权重
         有了每个hyp和每个hyp的权重之后有两种进化方式；
         1.根据每个hyp的权重随机选择一个之前的hyp作为base hyp，random.choices(range(n), weights=w)
         2.根据每个hyp的权重对之前所有的hyp进行融合获得一个base hyp，(x * w.reshape(n, 1)).sum(0) / w.sum()
         evolve.txt会记录每次进化之后的results+hyp
         每次进化时，hyp会根据之前的results进行从大到小的排序；
         再根据fitness函数计算之前每次进化得到的hyp的权重
         再确定哪一种进化方式，从而进行进化
         """
        for _ in range(100):  # 迭代演化
            if os.path.exists('evolve.txt'):  # 如果存在volve.txt：选择最佳的提示并演化
                # Select parent(s)
                parent = 'single'  # 父级选择方法：“单”或“加权”
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # 要考虑的先前结果数
                x = x[np.argsort(-fitness(x))][:n]  # 前n个突变
                w = fitness(x) - fitness(x).min()  # 权重
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # 选择权重
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 权重结合

                mp, s = 0.9, 0.2
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # 变异直到发生更改（防止重复）
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])

            # 修剪hyp在规定范围内
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])
                hyp[k] = min(hyp[k], v[2])
                hyp[k] = round(hyp[k], 5)

            # 训练
            results = train(hyp.copy(), opt, device)
            """
             写入results和对应的hyp到evolve.txt
             evolve.txt文件每一行为一次进化的结果
             一行中前七个数字为(P, R, mAP, F1, test_losses=(GIoU, obj, cls))，之后为hyp
             保存hyp到yaml文件
             """
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # 绘制结果
        plot_evolution(yaml_file)
        print('Hyperparameter evolution complete. Best results saved as: %s\nCommand to train a new model with these '
              'hyperparameters: $ python train.py --hyp %s' % (yaml_file, yaml_file))
