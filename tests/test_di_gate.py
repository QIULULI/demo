import pytest  # 中文注释：引入pytest以便按需跳过缺失依赖的测试

torch = pytest.importorskip('torch')  # 中文注释：若缺少PyTorch则跳过整份测试
nn = pytest.importorskip('torch.nn')  # 中文注释：按需导入nn模块供伪造特征提取器使用
F = pytest.importorskip('torch.nn.functional')  # 中文注释：按需导入函数式API生成网格特征

from mmengine.structures import InstanceData  # 中文注释：导入InstanceData以构建伪标签实例
from mmdet.structures import DetDataSample  # 中文注释：导入DetDataSample以模拟检测数据样本
from mmdet.models.detectors.Z_diff_semi_base import SemiBaseDiffDetector  # 中文注释：导入待测半监督检测器基类


class DummyFeatureExtractor(nn.Module):  # 中文注释：定义简化的特征提取器用于测试
    def __init__(self):  # 中文注释：初始化提取器
        super().__init__()  # 中文注释：调用父类初始化
        self.ssdc_feature_cache = {'noref': {}}  # 中文注释：准备缓存字典以模拟域不变特征

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:  # 中文注释：实现特征提取逻辑
        self.ssdc_feature_cache['noref'] = {'inv': (inputs.clone(),)}  # 中文注释：直接使用输入张量作为域不变特征缓存
        return inputs  # 中文注释：返回原始输入作为特征图


def _build_teacher_sample(origin_box, view_box, homography, ori_shape, img_shape):  # 中文注释：辅助函数创建教师视角数据样本
    sample = DetDataSample()  # 中文注释：实例化数据样本
    instances = InstanceData()  # 中文注释：创建实例数据容器
    tensor_origin = torch.tensor([origin_box], dtype=torch.float32)  # 中文注释：构造原图坐标框
    tensor_view = torch.tensor([view_box], dtype=torch.float32)  # 中文注释：构造教师视角框
    instances.bboxes = tensor_origin.clone()  # 中文注释：写入原图坐标到默认字段
    instances.origin_bboxes = tensor_origin.clone()  # 中文注释：缓存原图坐标副本
    instances.teacher_view_bboxes = tensor_view.clone()  # 中文注释：记录教师视角框
    sample.gt_instances = instances  # 中文注释：将实例数据绑定到样本
    sample.ori_shape = ori_shape  # 中文注释：记录原图尺寸
    sample.img_shape = img_shape  # 中文注释：记录教师输入尺寸
    sample.homography_matrix = torch.tensor(homography, dtype=torch.float32)  # 中文注释：写入单应性矩阵
    return sample  # 中文注释：返回构建好的样本


def _build_student_sample(view_box, origin_box, homography, ori_shape, img_shape):  # 中文注释：辅助函数创建学生视角数据样本
    sample = DetDataSample()  # 中文注释：实例化数据样本
    instances = InstanceData()  # 中文注释：创建实例数据容器
    tensor_view = torch.tensor([view_box], dtype=torch.float32)  # 中文注释：构造学生视角框
    tensor_origin = torch.tensor([origin_box], dtype=torch.float32)  # 中文注释：构造原图框
    instances.bboxes = tensor_view.clone()  # 中文注释：写入学生视角坐标
    instances.student_view_bboxes = tensor_view.clone()  # 中文注释：缓存学生输入坐标系下的框
    instances.origin_bboxes = tensor_origin.clone()  # 中文注释：记录原图坐标以便回投影
    sample.gt_instances = instances  # 中文注释：绑定实例数据
    sample.ori_shape = ori_shape  # 中文注释：记录原图尺寸
    sample.img_shape = img_shape  # 中文注释：记录学生输入尺寸
    sample.homography_matrix = torch.tensor(homography, dtype=torch.float32)  # 中文注释：写入学生单应性矩阵
    return sample  # 中文注释：返回学生样本


def _build_detector():  # 中文注释：构造仅包含所需属性的检测器实例
    detector = SemiBaseDiffDetector.__new__(SemiBaseDiffDetector)  # 中文注释：跳过繁琐初始化直接创建实例
    detector.teacher = DummyFeatureExtractor()  # 中文注释：注入教师特征提取器
    detector.student = DummyFeatureExtractor()  # 中文注释：注入学生特征提取器
    detector.semi_train_cfg = {'consistency_gate': 0.9}  # 中文注释：提供默认半监督配置避免访问错误
    return detector  # 中文注释：返回配置好的检测器


def test_di_gate_aligns_without_geometric_transform():  # 中文注释：验证无几何变换时余弦相似度足够高不会被过滤
    detector = _build_detector()  # 中文注释：获取测试用检测器
    identity_h = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]  # 中文注释：定义单位单应性矩阵
    teacher_sample = _build_teacher_sample([1., 1., 3., 3.], [1., 1., 3., 3.], identity_h, (4, 4), (4, 4))  # 中文注释：构造教师样本
    student_sample = _build_student_sample([1., 1., 3., 3.], [1., 1., 3., 3.], identity_h, (4, 4), (4, 4))  # 中文注释：构造学生样本
    teacher_inputs = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)  # 中文注释：使用网格值构造教师输入特征
    student_inputs = teacher_inputs.clone()  # 中文注释：学生输入与教师一致确保理想对齐
    ssdc_cfg = {'consistency_gate': 0.95}  # 中文注释：设置严格阈值以验证相似度接近1
    detector._apply_di_gate([teacher_sample], [student_sample], teacher_inputs, student_inputs, ssdc_cfg, current_iter=0)  # 中文注释：执行域不变门控
    assert student_sample.gt_instances.bboxes.shape[0] == 1  # 中文注释：确认伪框未被错误过滤


def test_di_gate_handles_scaling_homography():  # 中文注释：验证存在几何变换时依旧能够保持正确匹配
    detector = _build_detector()  # 中文注释：获取测试用检测器
    teacher_origin = [2., 2., 6., 6.]  # 中文注释：定义原图坐标框
    teacher_view = [1., 1., 3., 3.]  # 中文注释：定义教师视角框
    scale_h = [[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 1.]]  # 中文注释：构造缩放单应性矩阵
    identity_h = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]  # 中文注释：复用单位矩阵表示学生视角
    teacher_sample = _build_teacher_sample(teacher_origin, teacher_view, scale_h, (8, 8), (4, 4))  # 中文注释：构造教师样本
    student_sample = _build_student_sample([2., 2., 6., 6.], teacher_origin, identity_h, (8, 8), (8, 8))  # 中文注释：构造学生样本
    student_inputs = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8)  # 中文注释：学生使用高分辨率网格特征
    teacher_inputs = F.interpolate(student_inputs, size=(4, 4), mode='bilinear', align_corners=False)  # 中文注释：教师特征通过双线性插值模拟缩放视角
    ssdc_cfg = {'consistency_gate': 0.9}  # 中文注释：设置较高阈值防止误通过
    detector._apply_di_gate([teacher_sample], [student_sample], teacher_inputs, student_inputs, ssdc_cfg, current_iter=0)  # 中文注释：执行域不变门控
    assert student_sample.gt_instances.bboxes.shape[0] == 1  # 中文注释：确保几何变换不会导致伪框被全部过滤
