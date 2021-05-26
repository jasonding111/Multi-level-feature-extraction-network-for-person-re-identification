from utils.logger import AverageMeter


def do_train(
        epoch,
        model,
        center_criterion,
        train_loader,
        optimizer,
        optimizer_center,
        loss_fn,
):
    model.train()
    #model = nn.DataParallel(model).cuda()
    model = model.cuda()

    loss_meter = AverageMeter()
    id_loss_meter = AverageMeter()
    g_id_loss_meter = AverageMeter()
    p_id_loss_meter = AverageMeter()
    tri_loss_meter = AverageMeter()

    for batch_idx, (imgs, pids) in enumerate(train_loader):
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        imgs, pids = imgs.cuda(), pids.cuda()
        output = model(imgs, pids)
        loss, id_loss, g_id_loss, part_id_loss, tri_loss = loss_fn(output, pids)
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / 0.0005)
        optimizer_center.step()

        loss_meter.update(loss.item(), imgs.shape[0])
        id_loss_meter.update(id_loss.item(), imgs.shape[0])
        g_id_loss_meter.update(g_id_loss.item(), imgs.shape[0])
        p_id_loss_meter.update(part_id_loss.item(), imgs.shape[0])
        tri_loss_meter.update(tri_loss.item(), imgs.shape[0])

        if (batch_idx + 1) % 10 == 0:
            print(
                "Epoch[{}] Iteration[{}/{}] Loss: {:.3f} id_loss: {:.3f} g_id_loss: {:.3f} part_id_loss: {:.3f} "
                "tri_loss: {:.3f}".format(
                    epoch, (batch_idx + 1), len(train_loader),
                    loss_meter.avg, id_loss_meter.avg, g_id_loss_meter.avg, p_id_loss_meter.avg,
                    tri_loss_meter.avg))
