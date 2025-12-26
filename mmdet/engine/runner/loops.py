# # Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import is_model_wrapper
from mmengine.runner import ValLoop

from mmdet.registry import LOOPS


@LOOPS.register_module()
class TeacherStudentValLoop(ValLoop):
    """Validation loop for semi-supervised models with multiple branches.

    In this repo, there can be multiple "teacher-like" branches:
      - EMA teacher:   model.teacher        (updated by AdaptiveTeacherHook)
      - Diff teacher:  model.diff_detector  (optional, can be updated by DiffTeacherHeadEMAHook)
      - Student:       model.student

    This loop evaluates all available branches and writes metrics with
    explicit prefixes:
      - ema_teacher/...
      - diff_teacher/...    (only if diff_detector exists)
      - student/...

    Backward compatibility:
      - also writes teacher/... as an alias of ema_teacher/... so existing
        configs like save_best=['teacher/coco/bbox_mAP_50', ...] keep working.
    """

    def run(self):
        """Launch validation for EMA teacher, optional diff teacher, and student."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        # unwrap model wrapper (DDP / DP) first
        model = self.runner.model
        if is_model_wrapper(model):
            model = model.module

        # unwrap domain adaptation wrapper (DomainAdaptationDetector) if present.
        # In this repo, DomainAdaptationDetector stores the real semi model in `.model`.
        # We only unwrap while the current object doesn't expose teacher/student yet.
        unwrap_guard = 0
        while hasattr(model, 'model') and not (hasattr(model, 'teacher') and hasattr(model, 'student')):
            model = model.model
            unwrap_guard += 1
            if unwrap_guard >= 5:
                break

        assert hasattr(model, 'semi_test_cfg'), (
            'TeacherStudentValLoop expects the model to have `semi_test_cfg`.')
        assert hasattr(model, 'teacher'), (
            'TeacherStudentValLoop expects the model to have `teacher`.')
        assert hasattr(model, 'student'), (
            'TeacherStudentValLoop expects the model to have `student`.')

        # keep original setting
        orig_predict_on = model.semi_test_cfg.get('predict_on', None)

        # Decide which branches to evaluate
        predict_list = []
        # EMA teacher (always)
        predict_list.append('teacher')
        # Diff teacher (optional)
        if hasattr(model, 'diff_detector') and getattr(model, 'diff_detector') is not None:
            predict_list.append('diff_detector')
        # Student (always)
        predict_list.append('student')

        # Explicit prefix mapping
        prefix_map = {
            'teacher': 'ema_teacher',
            'diff_detector': 'diff_teacher',
            'student': 'student',
        }

        multi_metrics = dict()

        for branch_name in predict_list:
            model.semi_test_cfg['predict_on'] = branch_name

            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)

            # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

            # 1) explicit prefix
            explicit_prefix = prefix_map.get(branch_name, branch_name)
            multi_metrics.update({f'{explicit_prefix}/{k}': v for k, v in metrics.items()})

            # 2) backward-compatible alias for EMA teacher
            #    keep `teacher/...` as alias of `ema_teacher/...`
            if branch_name == 'teacher':
                multi_metrics.update({f'teacher/{k}': v for k, v in metrics.items()})

        # restore original setting safely
        if orig_predict_on is None:
            # if it wasn't set before, remove to avoid leaving a None value
            model.semi_test_cfg.pop('predict_on', None)
        else:
            model.semi_test_cfg['predict_on'] = orig_predict_on

        self.runner.call_hook('after_val_epoch', metrics=multi_metrics)
        self.runner.call_hook('after_val')
