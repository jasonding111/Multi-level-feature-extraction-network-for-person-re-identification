from losses.softmax_loss import CrossEntropyLabelSmooth
from losses.center_loss import CenterLoss
from losses.triplet_loss import WeightedTripletLoss


def build_loss(num_classes):
    feat_dim = 2048
    triplet = WeightedTripletLoss()

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)
    print("label smooth on, num_classes:{}".format(num_classes))

    def loss_func(outputs, target):
        g_id_loss = id_loss_func(outputs[0], target)
        p1_id_loss = id_loss_func(outputs[2], target)
        part_id_loss = [id_loss_func(output, target) for output in outputs[3:5]]
        part_id_loss = sum(part_id_loss) / len(part_id_loss)
        part_id_loss = (p1_id_loss + part_id_loss) / 2
        id_loss = 0.5 * g_id_loss + 0.5 * part_id_loss

        tri_loss = triplet(outputs[1], target)[0]
        center_loss = center_criterion(outputs[1], target)
        id_rate = 0.5
        tri_rate = 1
        center_rate = 0.0005

        return id_rate * id_loss + tri_rate * tri_loss + center_rate * center_loss, \
               id_loss, g_id_loss, part_id_loss, tri_loss

    return loss_func, center_criterion
