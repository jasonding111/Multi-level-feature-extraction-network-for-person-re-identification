import datetime
import os
import sys
import time
import torch
import os.path as osp
from torch.backends import cudnn
from data.data_loader_build import build_data_loader
from losses.loss_build import build_loss
from model.model_build import build_model
from utils.infer import inference
from utils.logger import Logger
from utils.lr_scheduler import WarmupMultiStepLR
from utils.optimizer_build import build_optimizer
from utils.trainer import do_train

evaluate = False
save_dir = "./"
resume = ""


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    cudnn.benchmark = True


def main(args=None):
    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not available")

    if not evaluate:
        sys.stdout = Logger(osp.join(save_dir, 'log_train_test.txt'))
    else:
        sys.stdout = Logger(osp.join(save_dir, 'log_test.txt'))

    print("Currently using GPU {}".format("0"))
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    set_seed(3)
    if args is not None:
        data = args["dataset"]
    else:
        data = 'm'
    train_loader, query_loader, gallery_loader, num_query, num_classes, test_loader, dataset_name = build_data_loader(
        data)

    print("Initializing model: {}".format("MFEN"))
    model = build_model(num_classes, data)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    loss_func, center_criterion = build_loss(num_classes)
    optimizer, optimizer_center = build_optimizer(model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, [40, 70], 0.1, 0.01, 10)

    start_epoch = 1
    max_epoch = 140

    if resume:
        print("Loading checkpoint from '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    if evaluate:
        print("Evaluate only")
        inference(model, query_loader, gallery_loader, dataset_name)
        return 0

    start = time.time()
    train_time = 0

    print("==> Start training")
    for epoch in range(start_epoch, max_epoch + 1):
        start_time = time.time()

        do_train(
            epoch,
            model,
            center_criterion,
            train_loader,
            optimizer,
            optimizer_center,
            loss_func,
        )
        scheduler.step()
        time_epoch = round(time.time() - start_time)
        print("Epoch {} done. Time of epoch: {:.3f}[s]"
              .format(epoch, time_epoch))
        train_time += time_epoch
        state = {'state_dict': model.state_dict(), 'epoch': epoch}
        if epoch % 20 == 0:
            torch.save(state, os.path.join(save_dir, "MFEN" + '_{}.pth'.format(epoch)))
        if epoch == max_epoch:
            # if epoch % 20 == 0:
            print("==> Test")
            inference(model, query_loader, gallery_loader, dataset_name)

    elapsed = round(time.time() - start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if __name__ == '__main__':
    main()
