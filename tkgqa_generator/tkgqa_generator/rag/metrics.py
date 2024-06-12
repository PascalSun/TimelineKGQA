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
        """Calculate the reciprocal rank of a single result list."""
        rank = r["rank"]
        labels = r["labels"]
        for i, val in enumerate(rank):
            if val:
                return 1 / (i + 1)
                # return 1 / (int((i / labels)) + 1)
        return 0

    return sum(reciprocal_rank(r) for r in rs) / len(rs)
