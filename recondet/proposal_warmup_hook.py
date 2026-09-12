from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class ReconDetProposalWarmupHook(Hook):
    """Expose the runner epoch to ReconDet proposal grouping."""

    def before_train_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'set_proposal_epoch'):
            model.set_proposal_epoch(runner.epoch)
