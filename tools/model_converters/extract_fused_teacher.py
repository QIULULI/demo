# -*- coding: utf-8 -*-  # 中文注释：声明文件编码确保中文注释正常显示
"""工具脚本：将Stage1融合模型中的RGB学生分支提取为独立教师ckpt。"""  # 中文注释：模块文档字符串描述用途
import argparse  # 中文注释：引入argparse用于解析命令行参数
from typing import Dict, Tuple  # 中文注释：引入类型别名便于阅读与静态检查

import torch  # 中文注释：引入PyTorch用于处理权重张量
from mmengine.config import Config  # 中文注释：引入Config以读取配置文件
from mmengine.logging import print_log  # 中文注释：引入print_log统一日志格式
from mmengine.runner import CheckpointLoader  # 中文注释：引入CheckpointLoader兼容多种ckpt格式

from mmdet.registry import MODELS  # 中文注释：引入模型注册表以构建检测器
from mmdet.utils import register_all_modules  # 中文注释：引入注册函数以确保组件可被发现


def parse_args() -> argparse.Namespace:  # 中文注释：定义命令行参数解析函数
    """解析提取脚本的命令行参数。"""  # 中文注释：函数文档说明
    parser = argparse.ArgumentParser(  # 中文注释：创建参数解析器实例
        description='提取student_rgb分支并生成单分支扩散检测器ckpt')  # 中文注释：填写脚本描述
    parser.add_argument(  # 中文注释：添加Stage1 ckpt路径参数
        '--stage1-ckpt',  # 中文注释：参数名称
        default='rgb_fused1111.pth',  # 中文注释：默认输入文件名
        help='Stage1 训练得到的融合模型 ckpt 路径')  # 中文注释：参数用途说明
    parser.add_argument(  # 中文注释：添加输出ckpt路径参数
        '--output-ckpt',  # 中文注释：参数名称
        default='rgb_fused_teacher_only.pth',  # 中文注释：默认输出文件名
        help='提取后的单分支教师 ckpt 保存路径')  # 中文注释：参数用途说明
    parser.add_argument(  # 中文注释：添加部署配置路径参数
        '--deploy-config',  # 中文注释：参数名称
        default=None,  # 中文注释：默认不指定配置只做提取
        help='可选：用于构建单分支模型的配置文件路径，用于权重匹配校验')  # 中文注释：参数用途说明
    parser.add_argument(  # 中文注释：添加前向自检开关
        '--run-forward-check',  # 中文注释：参数名称
        action='store_true',  # 中文注释：布尔开关存在即为True
        help='若提供配置则执行一次extract_feat前向检查，用于静态验证算子连通性')  # 中文注释：参数用途说明
    parser.add_argument(  # 中文注释：添加日志级别参数
        '--log-level',  # 中文注释：参数名称
        default='INFO',  # 中文注释：默认日志级别
        choices=['INFO', 'WARNING', 'ERROR'],  # 中文注释：限制可选日志级别
        help='控制日志输出的严重程度')  # 中文注释：参数用途说明
    return parser.parse_args()  # 中文注释：返回解析结果


def load_raw_state_dict(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict]:  # 中文注释：加载原始ckpt并拆分state_dict与meta
    """加载ckpt返回(state_dict, meta)。"""  # 中文注释：函数文档
    checkpoint = CheckpointLoader.load_checkpoint(  # 中文注释：使用通用加载器兼容多源ckpt
        ckpt_path, map_location='cpu')  # 中文注释：映射到CPU避免GPU依赖
    state_dict: Dict[str, torch.Tensor]  # 中文注释：显式声明类型
    meta: Dict  # 中文注释：显式声明元信息类型
    if 'state_dict' in checkpoint:  # 中文注释：标准mmengine ckpt格式
        state_dict = checkpoint['state_dict']  # 中文注释：提取参数字典
        meta = checkpoint.get('meta', {})  # 中文注释：提取meta缺省为空
    else:  # 中文注释：兼容直接保存state_dict的情况
        state_dict = checkpoint  # 中文注释：直接视为参数字典
        meta = {}  # 中文注释：缺省元信息
    return state_dict, meta  # 中文注释：返回解析结果


def extract_student_branch(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # 中文注释：提取学生RGB分支权重
    """过滤student_rgb相关参数并去除前缀。"""  # 中文注释：函数文档
    extracted: Dict[str, torch.Tensor] = {}  # 中文注释：初始化输出字典
    candidate_prefixes = (  # 中文注释：罗列可能的前缀组合
        'student_rgb.',  # 中文注释：常见单机保存格式
        'module.student_rgb.',  # 中文注释：分布式保存格式
        'model.student_rgb.',  # 中文注释：部分runner封装格式
    )
    for key, value in state_dict.items():  # 中文注释：遍历原始参数
        matched_prefix = None  # 中文注释：用于记录匹配到的前缀
        for prefix in candidate_prefixes:  # 中文注释：逐个尝试匹配
            if key.startswith(prefix):  # 中文注释：当键以当前前缀开头
                matched_prefix = prefix  # 中文注释：记录匹配前缀
                break  # 中文注释：匹配后提前结束内部循环
        if matched_prefix is None:  # 中文注释：未匹配到学生前缀则跳过
            continue  # 中文注释：进入下一个键
        new_key = key[len(matched_prefix):]  # 中文注释：去除前缀得到单分支模型期望的键
        extracted[new_key] = value  # 中文注释：写入输出权重
    if not extracted:  # 中文注释：若未提取到任何参数则提示错误
        raise KeyError('未找到student_rgb前缀参数，请确认输入ckpt来自DualDiffFusionStage1学生分支。')  # 中文注释：详细报错
    return extracted  # 中文注释：返回提取结果


def build_and_validate(config_path: str,  # 中文注释：配置文件路径
                       state_dict: Dict[str, torch.Tensor],  # 中文注释：提取后的权重
                       run_forward: bool = False) -> None:  # 中文注释：是否执行前向自检
    """根据配置构建模型并检查权重映射与可选前向。"""  # 中文注释：函数文档
    register_all_modules()  # 中文注释：注册全部模块保证构建成功
    cfg = Config.fromfile(config_path)  # 中文注释：读取配置
    model = MODELS.build(cfg.model)  # 中文注释：实例化单分支扩散检测器
    load_info = model.load_state_dict(state_dict, strict=False)  # 中文注释：以非严格方式加载权重以获知缺失与多余键
    missing_keys, unexpected_keys = load_info  # 中文注释：解包加载信息
    if missing_keys:  # 中文注释：若存在缺失键
        print_log(f'缺失参数列表：{missing_keys}', logger='current', level='WARNING')  # 中文注释：记录警告
    if unexpected_keys:  # 中文注释：若存在多余键
        print_log(f'多余参数列表：{unexpected_keys}', logger='current', level='WARNING')  # 中文注释：记录警告
    if run_forward:  # 中文注释：当需要运行前向检查时
        model.eval()  # 中文注释：切换评估模式避免梯度计算
        diff_cfg = cfg.model.get('backbone', {}).get('diff_config', {})  # 中文注释：读取扩散配置字典
        resolution = diff_cfg.get('input_resolution', [224, 224])  # 中文注释：获取输入分辨率缺省为224
        height, width = int(resolution[0]), int(resolution[1])  # 中文注释：转换为整数
        dummy_input = torch.randn(1, 3, height, width)  # 中文注释：构造假输入
        with torch.no_grad():  # 中文注释：关闭梯度
            _ = model.extract_feat(dummy_input)  # 中文注释：执行特征提取验证连通性
        print_log('extract_feat 前向检查完成。', logger='current', level='INFO')  # 中文注释：输出检查结果


def main() -> None:  # 中文注释：脚本主入口
    args = parse_args()  # 中文注释：解析命令行参数
    print_log(f'加载Stage1 ckpt：{args.stage1_ckpt}', logger='current', level=args.log_level)  # 中文注释：记录输入路径
    raw_state_dict, meta = load_raw_state_dict(args.stage1_ckpt)  # 中文注释：加载原始权重与元信息
    print_log('开始提取student_rgb分支参数。', logger='current', level=args.log_level)  # 中文注释：提示提取流程
    student_state_dict = extract_student_branch(raw_state_dict)  # 中文注释：执行权重筛选
    new_checkpoint = dict(state_dict=student_state_dict, meta=meta)  # 中文注释：构造新ckpt
    torch.save(new_checkpoint, args.output_ckpt)  # 中文注释：保存新ckpt
    print_log(f'已保存教师专用ckpt至：{args.output_ckpt}', logger='current', level=args.log_level)  # 中文注释：记录保存结果
    if args.deploy_config is not None:  # 中文注释：若提供配置则执行校验
        print_log('检测到部署配置，开始构建模型进行映射校验。', logger='current', level=args.log_level)  # 中文注释：提示校验阶段
        build_and_validate(args.deploy_config, student_state_dict, args.run_forward_check)  # 中文注释：执行构建与可选前向
    print_log('ckpt提取流程完成。', logger='current', level=args.log_level)  # 中文注释：结束日志


if __name__ == '__main__':  # 中文注释：脚本入口保护
    main()  # 中文注释：调用主函数

# 小型自检示例（供REPL复制）：  # 中文注释：提示快速验证方式
# >>> python tools/model_converters/extract_fused_teacher.py --stage1-ckpt rgb_fused1111.pth \  # 中文注释：执行纯提取命令行前半段
# ...     --output-ckpt rgb_fused_teacher_only.pth  # 中文注释：执行纯提取命令行后半段
# >>> python tools/model_converters/extract_fused_teacher.py --stage1-ckpt rgb_fused1111.pth \  # 中文注释：执行含校验的命令行前半段
# ...     --output-ckpt rgb_fused_teacher_only.pth \  # 中文注释：执行含校验命令行的输出路径部分
# ...     --deploy-config configs/diff/fused_teacher_deploy.py \  # 中文注释：执行含校验命令行的配置路径部分
# ...     --run-forward-check  # 中文注释：执行含校验的命令行尾段