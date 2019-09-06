import pyro
import pyro.distributions as dist
import pyro.contrib.autoguide as ag

import torch
from torch.distributions import constraints
#  Note the default prefix is '_auto'

from contextlib import ExitStack


class AutoNormal(ag.AutoGuide):
    """
    This implementation of :class:`AutoGuide` uses Normal(0, 1) distributions
    to construct a guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.

    It should be equivalent to pyro.contrib.autoguide.AutoDiagonalNormal, but
    with more convenient names.  In AutoDiagonalNormal, if your model has N
    named parameters with dimensions k_i and sum k_i = D, you get a single
    vector of length D for your mean, and a single vec of length D for sigmas.
    This guide gives you N distinct normals that you can call by name.

    Usage::

        guide = AutoNormal(model)
        svi = SVI(model, guide, ...)
    """
    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base
        ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the
        # model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates()
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                loc_name = "{}_{}_{}".format(self.prefix, name, 'loc')
                scale_name = "{}_{}_{}".format(self.prefix, name, 'scale')
                loc_value = pyro.param(
                    loc_name,
                    lambda: torch.zeros(site["fn"]._batch_shape + site["fn"]._event_shape),
                    constraint=site["fn"].support
                )
                scale_value = pyro.param(
                    scale_name,
                    lambda: torch.ones(site["fn"]._batch_shape + site["fn"]._event_shape),
                    constraint=constraints.positive
                )

                result[name] = pyro.sample(
                    name,
                    dist.Normal(
                        loc_value, scale_value
                    )
                )
        return result
