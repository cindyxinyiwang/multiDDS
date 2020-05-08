#!/usr/bin/env python3

from .translation import TranslationTask 
from . import adversaries
from . import adversarial_criterion
from . import FairseqTask, register_task


@register_task("translate_adversarial")
class TranslateAdversarialTask(TranslationTask):
    """Extends `PytorchTranslateTask` to account for the adversarial criterion
    and the adversary"""

    def build_adversary(self, args, model):
        return adversaries.build_adversary(args, model, self)

    def build_adversarial_criterion(self, args):
        return adversarial_criterion.build_criterion(args, self)
