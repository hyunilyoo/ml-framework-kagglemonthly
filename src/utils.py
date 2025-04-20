def create_fold_mapping(total_folds: int) -> dict[int, list[int]]:
    fold_mapping = {}

    for i in range(total_folds):
        remaining_folds = list(range(total_folds))
        remaining_folds.pop(i)  # Remove current fold
        fold_mapping[i] = remaining_folds
    return fold_mapping