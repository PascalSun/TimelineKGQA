from tkgqa_generator.utils import get_logger

logger = get_logger(__name__)


def mean_reciprocal_rank(rs):
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
    rs (list of lists): List of results for each query. Each result is a list of binary values
                        (1 if the item is relevant, 0 otherwise).

    Returns:
    float: Mean Reciprocal Rank (MRR) score.
    """

    def reciprocal_rank(r):
        """
        Calculate the reciprocal rank of a single result list.
        """
        rank = r["rank"]
        labels = r["labels"]
        # if all rank = 0, then rr = 0
        if sum(rank) == 0:
            return 0
        rr = 0
        for i, val in enumerate(rank):
            if val:
                rr += int(i / labels)

        if sum(rank) < labels:
            # punishment for not all labels are in the top k
            # we will assume then rest are all rank 31
            rr += int(31 / labels) * (labels - sum(rank))
        rr = 1 / (rr + 1)
        # print(rr, r["rank"], r["labels"])
        return rr

    return sum(reciprocal_rank(r) for r in rs) / len(rs)


def hit_n(rs, n=1):
    """
    Calculate Hit@N.
    Args:
    rs (list of lists): List of results for each query. Each result is a list of binary values
                        (1 if the item is relevant, 0 otherwise).
    n (int): The maximum rank to consider a hit.

    """

    def hit_at_n(r):
        """Calculate the hit@n of a single result list.

        If n = 1, then it is equivalent to precision@1.

        simple, medium, complex must all hit at n
        """
        rank = r["rank"]
        labels = r["labels"]
        rank = rank[: labels * n]
        if sum(rank) == labels:
            return 1
        return 0

    return sum(hit_at_n(r) for r in rs) / len(rs)
