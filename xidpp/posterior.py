import xidpp
import numpy as np
from numpyro.diagnostics import summary
from numpyro.infer import MCMC

class Posterior():
    def __init__(self, samples, rhat, n_eff, divergences, src_id):
        self.samples = samples
        self.rhat = rhat
        self.n_eff = n_eff
        self.divergences = divergences
        self.src_id = src_id

    @classmethod
    def from_numpyro(cls, mcmc: MCMC, prior: xidpp.Prior):
        samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
        stats = summary(samples, group_by_chain=True)
        
        post = cls(
            samples = samples,
            rhat = {k: np.asarray(s["r_hat"]) for k, s in stats.items()},
            n_eff = {k: np.asarray(s["n_eff"]) for k, s in stats.items()},
            divergences = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]),
            src_id = np.asarray(prior.src_id)
        )

        return post

    def merge_chains(self, site):
        # (nchain, nsample) -> (nchain * nsample)
        # (nchain, nsample, nsrc) -> (nchain * nsample, nsrcsw)

        return self.samples[site].reshape(-1, *self.samples[site].shape[2:])



    