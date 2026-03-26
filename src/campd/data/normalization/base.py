import torch


class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X=None, mins=None, maxs=None, means=None, stds=None):
        self.X = X
        if self.X is not None:
            data = torch.nan_to_num(X, nan=10e14)
            self.mins = data.min(dim=0).values
            data = torch.nan_to_num(X, nan=-10e14)
            self.maxs = data.max(dim=0).values
        else:
            assert mins is not None and maxs is not None
            self.mins = mins
            self.maxs = maxs

        if means is not None and stds is not None:
            self.means = means
            self.stds = stds

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{torch.round(self.mins, decimals=2)}\n    +: {torch.round(self.maxs, decimals=2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()

    def normalize_delta(self, delta):
        """Normalize a delta (difference) value.

        When computing finite differences in normalized space, the offset
        component of the normalizer cancels out.  Only the *scale* matters::

            delta_norm = normalize(x + delta) - normalize(x)

        Subclasses should override this with the closed-form scale when
        available.  The default implementation uses a finite-difference
        approximation that works for any normalizer.
        """
        zero = torch.zeros_like(delta)
        return self.normalize(delta) - self.normalize(zero)

    def export(self):
        return dict(mins=self.mins, maxs=self.maxs)
