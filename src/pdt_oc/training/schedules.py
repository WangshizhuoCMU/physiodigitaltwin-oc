def lambda_warmup(
    epoch: int,
    total: int = 50,
    start: float = 0.1,
    end: float = 1.0,
) -> float:
    if total <= 0:
        return end
    if epoch <= 0:
        return start
    if epoch >= total:
        return end
    progress = epoch / total
    return start + (end - start) * progress
