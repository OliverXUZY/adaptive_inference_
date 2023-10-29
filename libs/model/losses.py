import torch
import torch.nn.functional as F
from torch.autograd import Function


class RankSort(Function):
    """
    Implementation of the RankSort loss from
    Rank & Sort Loss for Object Detection and Instance Segmentation
    Code modified from
    https://github.com/kemaloksuz/RankSortLoss/blob/main/mmdet/models/losses/ranking_losses.py
    """
    @staticmethod
    def forward(
        ctx,
        logits,               # input logits (before sigmoid or softmax)
        targets,              # soft targets
        loss_normalizer=None, # normalizer, default: # fg
        delta=0.5,            # hyperparamer of the loss function
        eps=1e-10             # to prevent overflow
    ):
        assert delta > 0
        classification_grads = torch.zeros_like(logits)

        # Filter fg logits
        fg_labels = (targets > 0.)
        fg_logits = logits[fg_labels]
        fg_targets = targets[fg_labels]
        fg_num = len(fg_logits)

        # Do not use bg with scores less than minimum fg logit
        # since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits) - delta
        relevant_bg_labels = (targets == 0) & (logits >= threshold_logit)
        relevant_bg_num = relevant_bg_labels.sum().item()

        relevant_bg_logits = logits[relevant_bg_labels]
        relevant_bg_grad = logits.new_zeros((len(relevant_bg_logits),) )
        sorting_error = logits.new_zeros((fg_num, ))
        ranking_error = logits.new_zeros((fg_num, ))
        fg_grad = logits.new_zeros((fg_num, ))

        # sort the fg logits
        order = torch.argsort(fg_logits)

        # ordered -> fg_num x 1 (col vector)
        ord_fg_logits = fg_logits[order][:, None]
        ord_fg_targets = fg_targets[order][:, None]

        # auto broadcasting -> row vectors (fg_num, fg_num) / (fg_num, relevant_bg_num)
        fg_relations = (
            fg_logits[None, :]
            - ord_fg_logits.expand(fg_num, fg_num)
        )
        bg_relations = (
            relevant_bg_logits[None, :]
            - ord_fg_logits.expand(fg_num, relevant_bg_num)
        )

        # clip the values
        fg_relations /= 2 * delta
        fg_relations += 0.5
        fg_relations.clamp_(min=0, max=1)

        bg_relations /= 2 * delta
        bg_relations += 0.5
        bg_relations.clamp_(min=0, max=1)

        # ranking, sum over the rows -> fg_num / relevant_bg_num
        rank_pos = torch.sum(fg_relations, dim=-1)
        fp_num = torch.sum(bg_relations, dim=-1)
        rank = rank_pos + fp_num

        # ranking error (Eq 7)
        ranking_error[order] = fp_num / rank

        # sorting error (Eq 7)
        current_sorting_error = torch.sum(
            fg_relations * (1 - fg_targets),
            dim=-1
        )
        current_sorting_error /= rank_pos

        # find slots in the target sorted order for every slot
        iou_relations = (
            fg_targets[None, :] >= ord_fg_targets.expand(fg_num, fg_num)
        )
        target_sorted_order = iou_relations * fg_relations

        # rank of each slot
        rank_pos_target = torch.sum(target_sorted_order, dim=-1)

        # compute target sorting error. (Eq. 8)
        target_sorting_error = torch.sum(
            target_sorted_order * (1 - fg_targets),
            dim=-1
        )
        target_sorting_error /= rank_pos_target

        # sorting error
        sorting_error[order] = current_sorting_error - target_sorting_error

        # grads for ranking error (identity update)
        fp_num_eps = (fp_num > eps).float()
        fg_grad[order] -= ranking_error[order] * fp_num_eps
        relevant_bg_grad += torch.sum(
            bg_relations * (
                fp_num_eps * (ranking_error[order] / fp_num.clamp(min=eps))
            )[:, None],
            dim=0
        )

        # find the positives that are misranked (the cause of the error)
        # these are the ones with smaller IoU but larger logits
        missorted_examples = (~ iou_relations) * fg_relations

        # denorm of sorting pmf
        sorting_pmf_denom = torch.sum(missorted_examples, -1)

        # grads for sorting error
        sorting_pmf_denom_eps = (sorting_pmf_denom > eps).float()
        fg_grad[order] -= sorting_error[order] * sorting_pmf_denom_eps
        fg_grad += torch.sum(
            missorted_examples * (
                sorting_pmf_denom_eps * (
                    sorting_error[order] / sorting_pmf_denom.clamp(min=eps)
                )
            )[:, None],
            dim=0
        )

        # normalizer
        if loss_normalizer == None:
            normalizer = fg_num
        else:
            normalizer = loss_normalizer
        # normalize grads
        classification_grads[fg_labels] = fg_grad / normalizer
        classification_grads[relevant_bg_labels] = relevant_bg_grad / normalizer
        ranking_loss = ranking_error.sum() / normalizer
        sorting_loss = sorting_error.sum() / normalizer

        ctx.save_for_backward(classification_grads)
        return ranking_loss, sorting_loss

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, = ctx.saved_tensors
        return g1 * out_grad1, None, None, None


def rs_loss_fn(
    logits,         # (bs, n), unnormalized logits
    target,         # (bs, n), soft targets ((0, 1] for positives)
    delta=0.5,
):
    assert logits.size() == target.size()
    
    rank_loss, sort_loss = RankSort.apply(logits, target, None, delta)
    return rank_loss, sort_loss


def bce_loss_fn(
    logits,         # (bs, n): unnormalized logits
    target,         # (bs, n): binary targets
):
    assert logits.size() == target.size()
    target = target.float()

    loss = F.binary_cross_entropy_with_logits(logits, target)
    return loss


def vae_loss_fn(
    logits,         # (bs, kb), predicted logits
    target,         # (bs, kb), target binary vectors
    mu,             # (bs, d), mean of Gaussian posterior
    log_var,        # (bs, d), log variance of Gaussian posterior
):
    assert logits.size() == target.size()
    target = target.float()

    recon_loss = \
        F.binary_cross_entropy_with_logits(logits, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = (recon_loss + kl_loss) / len(logits)
    return loss