from .base import *

from .supervise_vanilla_vae_discreteClfDecoder import *
from .supervise_vanilla_vae_regressionClfDecoder import *

vae_models = {
    "SuperviseVanillaVAE_regressionClfDecoder": SuperviseVanillaVAE_regressionClfDecoder,
    "SuperviseVanillaVAE_discreteClfDecoder": SuperviseVanillaVAE_discreteClfDecoder,
    }
