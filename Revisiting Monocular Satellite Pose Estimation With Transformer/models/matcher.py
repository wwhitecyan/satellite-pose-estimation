# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the
        predictions of the network

    For efficiency reasons, the targets don't include the no_object.
    Because of this, in general, there are more predictions than targets.
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_pts: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_pts = cost_pts
        assert cost_class != 0 or cost_pts != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_pts = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_pts = torch.cat([v["landmarks"] for v in targets])

        # Compute the classification cost.
        # Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching,
        # it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between poitns
        cost_pts = torch.cdist(out_pts, tgt_pts, p=1)

        # Final cost matrix
        C = self.cost_pts * cost_pts + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["landmarks"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for
            i, c in enumerate(C.split(sizes, -1))
        ]
        return [(
            torch.as_tensor(i, dtype=torch.int64),
            torch.as_tensor(j, dtype=torch.int64)
        ) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_pts=args.set_cost_pts)
