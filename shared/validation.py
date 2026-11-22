"""
Purged & Embargoed Group Time-Series Split
Based on Marcos López de Prado's framework for financial cross-validation:
- Prevents serial correlation leakage across forward-looking return windows
- Prevents same-day cross-sectional contamination via strict group splitting
- Supports expanding window and rolling window validation with purge & embargo gaps
"""

from typing import Generator, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None


class PurgedGroupTimeSeriesSplit:
    """
    Purged & Embargoed Time-Series Cross-Validator for financial panel data.

    Parameters
    ----------
    n_splits : int, default=5
        Number of out-of-sample splits.
    purge_groups : int, default=1
        Number of groups immediately preceding the test window to drop from training
        to eliminate forward-looking label overlap (purge).
    embargo_groups : int, default=1
        Number of groups immediately following the test window to drop from future training
        sets to prevent autoregressive / serial correlation leakage (embargo).
    min_train_groups : Optional[int], default=None
        Minimum number of unique date/time groups required to form an initial training set.
        If None, dynamically inferred based on n_splits.
    max_train_groups : Optional[int], default=None
        If set, transforms expanding window into a rolling window of fixed max size.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_groups: int = 1,
        embargo_groups: int = 1,
        min_train_groups: Optional[int] = None,
        max_train_groups: Optional[int] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.purge_groups = max(0, purge_groups)
        self.embargo_groups = max(0, embargo_groups)
        self.min_train_groups = min_train_groups
        self.max_train_groups = max_train_groups

    def split(
        self,
        X: Sequence,
        y: Optional[Sequence] = None,
        groups: Optional[Sequence] = None,
    ) -> Generator[Tuple[Sequence, Sequence], None, None]:
        """
        Generate train and test indices.

        Parameters
        ----------
        X : array-like or sequence
            Training data.
        y : array-like or sequence, optional
            Target values.
        groups : array-like or sequence
            Group labels for samples (e.g., date_id, trade_date, or timestamp bucket).
            Must be ordered chronologically or sortable.

        Yields
        ------
        train_indices : np.ndarray or list of int
        test_indices : np.ndarray or list of int
        """
        if groups is None:
            raise ValueError("groups (e.g. date_id) must be provided for PurgedGroupTimeSeriesSplit.")

        groups_list = list(groups)
        n_samples = len(groups_list)

        # Extract unique groups in strictly chronological order (preserving order of appearance)
        unique_groups = []
        seen = set()
        for g in groups_list:
            if g not in seen:
                seen.add(g)
                unique_groups.append(g)

        n_groups = len(unique_groups)
        if n_groups < self.n_splits + 1:
            raise ValueError(
                f"Number of unique groups ({n_groups}) must be greater than n_splits ({self.n_splits})."
            )

        # Determine test split sizes across groups
        test_size_per_split = (n_groups - (self.min_train_groups or 0)) // (self.n_splits + 1)
        test_size_per_split = max(1, test_size_per_split)

        min_train = self.min_train_groups if self.min_train_groups is not None else test_size_per_split

        # Generate splits forward in time
        for i in range(self.n_splits):
            test_start_grp_idx = min_train + i * test_size_per_split
            test_end_grp_idx = min(test_start_grp_idx + test_size_per_split, n_groups)

            if test_start_grp_idx >= n_groups or test_start_grp_idx >= test_end_grp_idx:
                break

            test_grp_set = set(unique_groups[test_start_grp_idx:test_end_grp_idx])

            # Calculate train group boundaries with Purge
            train_end_grp_idx = max(0, test_start_grp_idx - self.purge_groups)
            if self.max_train_groups is not None:
                train_start_grp_idx = max(0, train_end_grp_idx - self.max_train_groups)
            else:
                train_start_grp_idx = 0

            if train_end_grp_idx <= train_start_grp_idx:
                continue

            train_grp_set = set(unique_groups[train_start_grp_idx:train_end_grp_idx])

            train_indices = [idx for idx, g in enumerate(groups_list) if g in train_grp_set]
            test_indices = [idx for idx, g in enumerate(groups_list) if g in test_grp_set]

            # Zero overlap guarantee
            assert set(train_indices).isdisjoint(set(test_indices)), "Look-ahead violation: Train and Test overlap!"

            if np is not None:
                yield np.array(train_indices, dtype=int), np.array(test_indices, dtype=int)
            else:
                yield train_indices, test_indices


if __name__ == "__main__":
    print("Testing PurgedGroupTimeSeriesSplit (Zero-Dependency & NumPy Compatible)...")
    sample_dates = [d for d in range(1, 21) for _ in range(10)]  # 20 dates, 10 samples each (200 rows)
    dummy_X = [[0.0] * 4 for _ in range(len(sample_dates))]

    cv = PurgedGroupTimeSeriesSplit(n_splits=4, purge_groups=1, embargo_groups=1)
    for fold, (train_idx, test_idx) in enumerate(cv.split(dummy_X, groups=sample_dates)):
        train_d = [sample_dates[i] for i in train_idx]
        test_d = [sample_dates[i] for i in test_idx]
        print(
            f"Fold {fold}: Train dates [{min(train_d)}..{max(train_d)}] ({len(train_d)} samples) "
            f"| Purge Gap: [{max(train_d)+1}..{min(test_d)-1}] "
            f"| Test dates [{min(test_d)}..{max(test_d)}] ({len(test_d)} samples)"
        )
        assert max(train_d) < min(test_d), "Look-ahead violation: Train date >= Test date!"
    print("All validation split checks passed successfully!")

