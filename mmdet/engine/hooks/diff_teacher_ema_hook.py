# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""EMA update hook for *diff teacher* heads.

This hook is designed for the Stage-2 setting described in this repo where:
  - ``model.student`` is the only network optimized by back-propagation;
  - ``model.diff_detector`` (the fused diffusion teacher from Stage-1)
    participates in forward to provide pseudo labels / distillation targets;
  - Only the *heads* of ``diff_detector`` (e.g. RPN / ROI) are updated by EMA
    from the student, while the diffusion encoder / fused FPN remain frozen.

Compared with the existing ``AdaptiveTeacherHook`` (which maintains an EMA
copy of ``model.teacher``), this hook updates the real diffusion teacher.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.runner import load_checkpoint

from mmdet.registry import HOOKS


def _unwrap_runner_model(runner: Runner) -> nn.Module:
    """Unwrap DDP / ModelWrapper and DomainAdaptationDetector wrapper."""
    model = runner.model
    if is_model_wrapper(model):
        model = model.module
    # DomainAdaptationDetector keeps the real semi model in ``model.model``.
    if hasattr(model, 'model'):
        model = model.model
    return model


def _get_attr(obj: object, name: str) -> Optional[object]:
    return getattr(obj, name, None)


@HOOKS.register_module()
class DiffTeacherHeadEMAHook(Hook):
    """EMA update ``diff_detector`` heads from ``student``.

    Args:
        momentum (float): EMA momentum in
            ``teacher = (1-m) * teacher + m * student``.
        interval (int): Update interval (in iterations).
        burn_up_iters (int): Skip EMA updates before this iteration.
        target_modules (Sequence[str]): Submodule names to update.
            Defaults to ('rpn_head', 'roi_head').
        skip_buffers (bool): If True, only update parameters.
            If False, also update buffers (e.g. running_mean/var).
        init_from_student (bool): If True, do a one-time hard copy
            (momentum=1.0) at iter==0 for the target modules.
            Default False (keep Stage-1 teacher weights at start).
        update_all_diff_teachers (bool): If True, update every diffusion
            teacher in ``model.diff_detectors`` dict (if exists). Otherwise
            only update ``model.diff_detector``.
        strict (bool): If True, enforce exact parameter/buffer name matching
            between student and target teacher module. If False, mismatched
            keys are skipped with warnings.
    """

    priority = 'ABOVE_NORMAL'

    def __init__(
        self,
        momentum: float = 0.0004,
        interval: int = 1,
        burn_up_iters: int = 0,
        target_modules: Sequence[str] = ('rpn_head', 'roi_head'),
        skip_buffers: bool = True,
        init_from_student: bool = False,
        update_all_diff_teachers: bool = False,
        strict: bool = True,
    ) -> None:
        super().__init__()
        assert 0.0 < float(momentum) < 1.0
        self.momentum = float(momentum)
        self.interval = int(max(1, interval))
        self.burn_up_iters = int(max(0, burn_up_iters))
        self.target_modules = tuple(target_modules)
        self.skip_buffers = bool(skip_buffers)
        self.init_from_student = bool(init_from_student)
        self.update_all_diff_teachers = bool(update_all_diff_teachers)
        self.strict = bool(strict)

    def before_train(self, runner: Runner) -> None:
        model = _unwrap_runner_model(runner)
        assert hasattr(model, 'student'), 'DiffTeacherHeadEMAHook requires model.student'
        assert hasattr(model, 'diff_detector') or hasattr(model, 'diff_detectors'), (
            'DiffTeacherHeadEMAHook requires model.diff_detector or model.diff_detectors')

        # Optional: load student-only pretrained weights (kept for parity with
        # AdaptiveTeacherHook; stage-2 can decide to set it or not).
        semi_train_cfg = _get_attr(model, 'semi_train_cfg')
        student_pretrained = None
        if semi_train_cfg is not None:
            if hasattr(semi_train_cfg, 'get'):
                student_pretrained = semi_train_cfg.get('student_pretrained', None)
            else:
                student_pretrained = getattr(semi_train_cfg, 'student_pretrained', None)
        if student_pretrained:
            load_checkpoint(model.student, student_pretrained, map_location='cpu', strict=False)
            # keep behavior consistent with existing repo hooks
            if torch.cuda.is_available():
                model.student.cuda()

        # Optional: hard copy at initial stage.
        if runner.iter == 0 and self.init_from_student:
            self._ema_update(model, momentum=1.0, runner=runner)

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[dict] = None,
    ) -> None:
        # Burn-in: keep the Stage-1 diff teacher unchanged during student warm-up.
        if runner.iter < self.burn_up_iters:
            return

        # Update schedule: apply interval only after burn-up.
        if ((runner.iter - self.burn_up_iters) + 1) % self.interval != 0:
            return

        model = _unwrap_runner_model(runner)
        self._ema_update(model, momentum=self.momentum, runner=runner)

    @torch.no_grad()
    def _ema_update(self, model: nn.Module, momentum: float, runner: Optional[Runner] = None) -> None:
        """Apply EMA updates for configured target modules."""
        student = model.student

        # Choose which diffusion teacher(s) to update.
        teacher_list = []
        if self.update_all_diff_teachers and isinstance(getattr(model, 'diff_detectors', None), dict):
            teacher_list = [t for t in model.diff_detectors.values() if t is not None]
        elif hasattr(model, 'diff_detector') and getattr(model, 'diff_detector', None) is not None:
            teacher_list = [model.diff_detector]
        else:
            # Nothing to update.
            return

        for teacher in teacher_list:
            for module_name in self.target_modules:
                stu_mod = getattr(student, module_name, None)
                tea_mod = getattr(teacher, module_name, None)
                if tea_mod is None and hasattr(teacher, 'student_rgb'):
                    tea_mod = getattr(teacher.student_rgb, module_name, None)                
                if stu_mod is None or tea_mod is None:
                    # In strict mode we raise to expose wrong assumptions early.
                    if self.strict:
                        raise AttributeError(
                            f'DiffTeacherHeadEMAHook cannot find module "{module_name}" '
                            f'on student={type(student)} or teacher={type(teacher)}')
                    self._log(runner, f'[DiffTeacherHeadEMAHook] skip: missing {module_name}', level='warning')
                    continue
                self._ema_update_module(stu_mod, tea_mod, momentum, runner=runner)

    @torch.no_grad()
    def _ema_update_module(
        self,
        student_module: nn.Module,
        teacher_module: nn.Module,
        momentum: float,
        runner: Optional[Runner] = None,
    ) -> None:
        """EMA update parameters (and optional buffers) of one submodule."""
        # Update parameters by name to be robust to ordering.
        stu_params = dict(student_module.named_parameters())
        tea_params = dict(teacher_module.named_parameters())

        if self.strict:
            missing = set(tea_params.keys()) - set(stu_params.keys())
            if missing:
                raise KeyError(
                    f'EMA strict mode: teacher has params not in student: {sorted(list(missing))[:20]}')

        for name, tea_p in tea_params.items():
            if name not in stu_params:
                self._log(runner, f'[DiffTeacherHeadEMAHook] param missing in student: {name}', level='warning')
                continue
            stu_p = stu_params[name]
            if tea_p.data.shape != stu_p.data.shape:
                msg = (f'[DiffTeacherHeadEMAHook] shape mismatch for {name}: '
                       f'teacher {tuple(tea_p.shape)} vs student {tuple(stu_p.shape)}')
                if self.strict:
                    raise RuntimeError(msg)
                self._log(runner, msg, level='warning')
                continue
            tea_p.data.mul_(1.0 - momentum).add_(stu_p.data, alpha=momentum)

        if self.skip_buffers:
            return

        # Update buffers (BN running stats etc.) by name.
        stu_bufs = dict(student_module.named_buffers())
        tea_bufs = dict(teacher_module.named_buffers())

        if self.strict:
            missing_buf = set(tea_bufs.keys()) - set(stu_bufs.keys())
            if missing_buf:
                raise KeyError(
                    f'EMA strict mode: teacher has buffers not in student: {sorted(list(missing_buf))[:20]}')

        for name, tea_b in tea_bufs.items():
            if name not in stu_bufs:
                self._log(runner, f'[DiffTeacherHeadEMAHook] buffer missing in student: {name}', level='warning')
                continue
            stu_b = stu_bufs[name]
            if tea_b.data.shape != stu_b.data.shape:
                msg = (f'[DiffTeacherHeadEMAHook] buffer shape mismatch for {name}: '
                       f'teacher {tuple(tea_b.shape)} vs student {tuple(stu_b.shape)}')
                if self.strict:
                    raise RuntimeError(msg)
                self._log(runner, msg, level='warning')
                continue
            if tea_b.dtype.is_floating_point:
                tea_b.data.mul_(1.0 - momentum).add_(stu_b.data, alpha=momentum)
            else:
                tea_b.data.copy_(stu_b.data)

    @staticmethod
    def _log(runner: Optional[Runner], msg: str, level: str = 'info') -> None:
        logger = getattr(runner, 'logger', None) if runner is not None else None
        if logger is None:
            return
        log_fn = getattr(logger, level, None)
        if callable(log_fn):
            log_fn(msg)
        else:
            logger.info(msg)
