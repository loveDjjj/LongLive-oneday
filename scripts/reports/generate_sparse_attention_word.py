#!/usr/bin/env python3
"""生成 LongLive2.0 稀疏注意力性能与质量实验 Word 报告。"""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/experiments/longlive_sparse_attention_results.docx"
RED = RGBColor(0xC0, 0x00, 0x00)
BLUE = "1F4E78"
LIGHT_BLUE = "D9EAF7"
LIGHT_GRAY = "F2F2F2"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, value, *, bold=False, color=None, size=8.0):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(str(value))
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    if color is not None:
        run.font.color.rgb = color
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_margins(cell, top=55, start=65, bottom=55, end=65):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_caption(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(3)
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(10)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")


def add_table(document, caption, headers, rows, highlights=None, font_size=8.0):
    add_caption(document, caption)
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table.autofit = True
    highlights = highlights or set()

    header = table.rows[0]
    set_repeat_table_header(header)
    for col, value in enumerate(headers):
        set_cell_text(header.cells[col], value, bold=True, color=RGBColor(255, 255, 255), size=font_size)
        set_cell_shading(header.cells[col], BLUE)
        set_cell_margins(header.cells[col])

    for row_index, values in enumerate(rows):
        row = table.add_row()
        for col_index, value in enumerate(values):
            key = (row_index, col_index)
            set_cell_text(
                row.cells[col_index],
                value,
                bold=key in highlights,
                color=RED if key in highlights else None,
                size=font_size,
            )
            if row_index % 2:
                set_cell_shading(row.cells[col_index], LIGHT_GRAY)
            if key in highlights:
                set_cell_shading(row.cells[col_index], LIGHT_BLUE)
            set_cell_margins(row.cells[col_index])
    return table


def add_body(document, text, *, bold=False, color=None):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.line_spacing = 1.15
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.size = Pt(9.5)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    if color is not None:
        run.font.color.rgb = color
    return paragraph


def add_bullet(document, text, *, key=False):
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(text)
    run.font.size = Pt(9.5)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    if key:
        run.bold = True
        run.font.color.rgb = RED


def configure_document(document):
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Cm(29.7)
    section.page_height = Cm(21.0)
    section.top_margin = Cm(1.25)
    section.bottom_margin = Cm(1.25)
    section.left_margin = Cm(1.25)
    section.right_margin = Cm(1.25)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    normal.font.size = Pt(9.5)
    for style_name, size, color in (
        ("Title", 22, RGBColor(0x1F, 0x1F, 0x1F)),
        ("Heading 1", 16, RGBColor(0x1F, 0x4E, 0x78)),
        ("Heading 2", 12, RGBColor(0x2F, 0x55, 0x96)),
    ):
        style = styles[style_name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        style.font.size = Pt(size)
        style.font.color.rgb = color
        style.font.bold = True

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("LongLive2.0 稀疏注意力实验报告  |  2026-08-14")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0x70, 0x70, 0x70)


def add_title(document):
    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("LongLive2.0 稀疏注意力实验结果")
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("性能数据与 VBench 质量数据汇总")
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    add_body(document, "报告日期：2026-08-14    设备平台：昇腾 NPU    计算精度：BF16")
    add_body(document, "说明：红色粗体表示关键最优值、关键退化或需要重点解释的结果。")


def add_performance_section(document):
    document.add_heading("第一部分  性能结果", level=1)
    add_table(
        document,
        "表 1  性能实验配置",
        ["项目", "配置"],
        [
            ["模型与权重", "LongLive2.0-5B 基础权重"],
            ["稀疏后端", "MindIE-SD RainFusion（推理）"],
            ["方法", "dense / hsa_cag / sla_cag / hsa_sla_cag"],
            ["视频长度", "5 s / 32 s / 64 s（125 / 765 / 1533 pixel frames）"],
            ["并行布局", "SP1×DP1 或 SP4×DP1"],
            ["统计口径", "无 profiler；预热后 3 次有效测量；报告 p50"],
            ["异步 VAE", "1 张专用 NPU；DiT 与分块 VAE 解码流水重叠"],
        ],
        font_size=8.5,
    )

    add_table(
        document,
        "表 2  DiT-only 延迟与相对 dense 加速比（SP1，p50）",
        ["方法", "5 s 延迟 / 加速比", "32 s 延迟 / 加速比", "64 s 延迟 / 加速比"],
        [
            ["dense", "18.372 s / 1.000×", "133.382 s / 1.000×", "271.492 s / 1.000×"],
            ["hsa_cag", "15.741 s / 1.167×", "100.933 s / 1.321×", "204.017 s / 1.331×"],
            ["hsa_sla_cag", "13.890 s / 1.323×", "85.927 s / 1.552×", "172.847 s / 1.571×"],
            ["sla_cag", "13.703 s / 1.341×", "84.056 s / 1.587×", "169.577 s / 1.601×"],
        ],
        highlights={(3, 1), (3, 2), (3, 3)},
    )

    add_table(
        document,
        "表 3  DiT-only 延迟与相对 dense 加速比（SP4，p50）",
        ["方法", "5 s 延迟 / 加速比", "32 s 延迟 / 加速比", "64 s 延迟 / 加速比"],
        [
            ["dense", "7.195 s / 1.000×", "49.192 s / 1.000×", "99.713 s / 1.000×"],
            ["hsa_cag", "6.519 s / 1.104×", "40.421 s / 1.217×", "80.971 s / 1.231×"],
            ["hsa_sla_cag", "6.230 s / 1.155×", "38.799 s / 1.268×", "78.735 s / 1.266×"],
            ["sla_cag", "6.123 s / 1.175×", "38.135 s / 1.290×", "76.040 s / 1.311×"],
        ],
        highlights={(3, 1), (3, 2), (3, 3)},
    )

    add_table(
        document,
        "表 4  32 s DiT-only 的 SP1→SP4 扩展效率",
        ["方法", "SP1 p50", "SP4 p50", "扩展加速比", "4 卡并行效率"],
        [
            ["dense", "133.382 s", "49.192 s", "2.711×", "67.78%"],
            ["hsa_cag", "100.933 s", "40.421 s", "2.497×", "62.42%"],
            ["hsa_sla_cag", "85.927 s", "38.799 s", "2.215×", "55.37%"],
            ["sla_cag", "84.056 s", "38.135 s", "2.204×", "55.11%"],
        ],
        highlights={(0, 3), (0, 4)},
    )

    add_table(
        document,
        "表 5  异步 VAE 完整视频端到端延迟（p50）",
        ["并行", "方法", "5 s", "32 s", "64 s", "64 s 相对 dense"],
        [
            ["SP1", "dense", "23.304 s", "138.555 s", "280.811 s", "1.000×"],
            ["SP1", "hsa_cag", "22.833 s", "120.053 s", "238.627 s", "1.177×"],
            ["SP1", "hsa_sla_cag", "22.822 s", "122.264 s", "236.810 s", "1.186×"],
            ["SP1", "sla_cag", "22.890 s", "121.757 s", "237.568 s", "1.182×"],
            ["SP4", "dense", "21.004 s", "119.962 s", "239.501 s", "1.000×"],
            ["SP4", "hsa_cag", "21.334 s", "124.013 s", "239.654 s", "0.999×"],
            ["SP4", "hsa_sla_cag", "21.528 s", "120.728 s", "240.105 s", "0.997×"],
            ["SP4", "sla_cag", "21.389 s", "120.533 s", "241.495 s", "0.992×"],
        ],
        highlights={(2, 4), (2, 5), (7, 4), (7, 5)},
    )

    add_table(
        document,
        "表 6  SP4 异步 VAE 阶段分解（3 次均值）",
        ["长度", "方法", "端到端", "AR loop", "VAE decode", "VAE drain"],
        [
            ["32 s", "dense", "120.054 s", "104.640 s", "114.557 s", "11.360 s"],
            ["32 s", "hsa_cag", "123.497 s", "58.584 s", "116.729 s", "59.677 s"],
            ["32 s", "hsa_sla_cag", "121.511 s", "57.384 s", "116.159 s", "60.246 s"],
            ["32 s", "sla_cag", "121.775 s", "57.284 s", "116.044 s", "60.241 s"],
            ["64 s", "dense", "241.419 s", "219.532 s", "229.463 s", "11.369 s"],
            ["64 s", "hsa_cag", "240.230 s", "114.780 s", "230.866 s", "117.594 s"],
            ["64 s", "hsa_sla_cag", "240.759 s", "114.719 s", "230.868 s", "117.642 s"],
            ["64 s", "sla_cag", "241.125 s", "114.614 s", "230.719 s", "117.606 s"],
        ],
        highlights={(3, 3), (7, 3), (7, 5)},
    )

    document.add_heading("性能结论", level=2)
    add_bullet(document, "稀疏注意力在 DiT-only 上有效：SP1、64 s 下 SLA+CAG 达到 1.601×，HSA+SLA+CAG 达到 1.571×。", key=True)
    add_bullet(document, "SP4 下 DiT 收益仍存在，但并行效率下降；64 s SLA+CAG 的 DiT 加速比为 1.311×。")
    add_bullet(document, "异步 VAE 端到端被 VAE 解码和 drain 主导；SP4、64 s 下 SLA+CAG 反而为 0.992×，不能用算子或 DiT-only 加速比代表整链路收益。", key=True)


def add_quality_section(document):
    document.add_page_break()
    document.add_heading("第二部分  VBench 质量结果", level=1)
    add_table(
        document,
        "表 7  质量实验配置与比较约束",
        ["项目", "配置"],
        [
            ["评测集", "LongLive2.0 标准 VBench 5% 子集，125 pixel frames"],
            ["基础权重", "longlive2_merged_generator.pt"],
            ["微调权重", "SLA+CAG 200-step；HSA+SLA+CAG 200-step"],
            ["训练范围", "两者统一为主干 LoRA + 原始 sla_linear"],
            ["报告指标", "VBench Quality / Semantic / Total 及 16 个子维度"],
            ["比较原则", "同一权重内 dense 与匹配稀疏路由优先；跨运行结论需核对 manifest"],
        ],
        font_size=8.5,
    )

    add_table(
        document,
        "表 8  基础权重的稀疏方法质量",
        ["推理方法", "Quality", "Semantic", "Total", "Δ Total vs dense"],
        [
            ["dense", "85.54", "71.99", "82.83", "0.00"],
            ["hsa_cag", "85.43", "72.26", "82.80", "-0.03"],
            ["sla_cag", "80.63", "66.43", "77.79", "-5.04"],
            ["hsa_sla_cag", "80.49", "66.79", "77.75", "-5.08"],
        ],
        highlights={(1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)},
    )

    add_table(
        document,
        "表 9  200-step 权重 × 推理路由的 2×3 总分矩阵",
        ["训练权重", "推理路由", "Quality", "Semantic", "Total", "相对同权重 dense"],
        [
            ["SLA+CAG 200-step", "dense", "85.61", "71.29", "82.75", "0.00"],
            ["SLA+CAG 200-step", "sla_cag", "81.73", "70.33", "79.45", "-3.30"],
            ["SLA+CAG 200-step", "hsa_sla_cag", "81.14", "68.93", "78.70", "-4.05"],
            ["Hybrid 200-step", "dense", "85.81", "70.88", "82.82", "0.00"],
            ["Hybrid 200-step", "sla_cag", "81.06", "68.66", "78.58", "-4.24"],
            ["Hybrid 200-step", "hsa_sla_cag", "82.22", "69.65", "79.71", "-3.12"],
        ],
        highlights={(1, 4), (1, 5), (5, 4), (5, 5)},
    )

    add_table(
        document,
        "表 10  匹配路由相对同权重 dense 的关键子指标残差",
        ["匹配方案", "Subject", "Background", "Aesthetic", "Imaging", "Object", "Multiple", "Spatial", "Dynamic", "Total"],
        [
            ["SLA 权重 + SLA 路由", "-5.85", "-5.66", "-2.10", "-3.26", "-7.08", "0.00", "-9.03", "-6.67", "-3.30"],
            ["Hybrid 权重 + Hybrid 路由", "-6.18", "-6.88", "-1.92", "-2.06", "-12.08", "-16.25", "-15.41", "0.00", "-3.12"],
        ],
        highlights={(0, 8), (0, 9), (1, 5), (1, 6), (1, 7), (1, 9)},
        font_size=7.5,
    )

    add_table(
        document,
        "表 11  200-step 微调对基础稀疏质量缺口的恢复",
        ["方法", "基础权重稀疏 Total", "200-step 匹配路由 Total", "绝对恢复", "归一化恢复率"],
        [
            ["sla_cag", "77.79", "79.45", "+1.66", "34.6%"],
            ["hsa_sla_cag", "77.75", "79.71", "+1.96", "38.6%"],
        ],
        highlights={(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)},
    )

    add_table(
        document,
        "表 12  路由专门化交叉比较",
        ["训练权重", "SLA 路由 Total", "Hybrid 路由 Total", "Hybrid - SLA"],
        [
            ["SLA+CAG 200-step", "79.45", "78.70", "-0.75"],
            ["Hybrid 200-step", "78.58", "79.71", "+1.12"],
            ["交互效应", "—", "—", "1.88"],
        ],
        highlights={(0, 1), (1, 2), (2, 3)},
    )

    add_table(
        document,
        "表 13  SLA+CAG 200-step 权重的 16 个 VBench 子指标",
        ["指标", "dense", "sla_cag", "hsa_sla_cag", "匹配稀疏 - dense"],
        [
            ["Subject consistency", "95.85", "90.00", "91.70", "-5.85"],
            ["Background consistency", "94.19", "88.53", "89.75", "-5.66"],
            ["Aesthetic quality", "60.72", "58.63", "58.25", "-2.10"],
            ["Imaging quality", "67.20", "63.94", "63.15", "-3.26"],
            ["Object class", "77.08", "70.00", "78.33", "-7.08"],
            ["Multiple objects", "48.44", "48.44", "49.38", "0.00"],
            ["Color", "83.31", "86.32", "80.10", "+3.01"],
            ["Spatial relationship", "88.30", "79.26", "74.66", "-9.03"],
            ["Scene", "41.25", "41.25", "31.56", "0.00"],
            ["Temporal style", "25.46", "25.84", "25.80", "+0.38"],
            ["Overall consistency", "23.26", "23.65", "23.81", "+0.39"],
            ["Human action", "96.00", "96.00", "96.00", "0.00"],
            ["Temporal flickering", "99.59", "98.79", "98.83", "-0.80"],
            ["Motion smoothness", "98.51", "98.54", "98.61", "+0.04"],
            ["Dynamic degree", "93.33", "86.67", "73.33", "-6.67"],
            ["Appearance style", "18.43", "19.10", "19.21", "+0.67"],
        ],
        highlights={(0, 4), (1, 4), (4, 4), (7, 4), (14, 4)},
        font_size=7.5,
    )

    add_table(
        document,
        "表 14  HSA+SLA+CAG 200-step 权重的 16 个 VBench 子指标",
        ["指标", "dense", "sla_cag", "hsa_sla_cag", "匹配稀疏 - dense"],
        [
            ["Subject consistency", "96.64", "90.83", "90.46", "-6.18"],
            ["Background consistency", "95.29", "88.36", "88.41", "-6.88"],
            ["Aesthetic quality", "60.30", "57.78", "58.38", "-1.92"],
            ["Imaging quality", "65.65", "62.65", "63.59", "-2.06"],
            ["Object class", "80.42", "70.42", "68.33", "-12.08"],
            ["Multiple objects", "61.25", "58.44", "45.00", "-16.25"],
            ["Color", "82.50", "78.57", "84.46", "+1.96"],
            ["Spatial relationship", "92.75", "79.10", "77.34", "-15.41"],
            ["Scene", "22.81", "23.12", "41.88", "+19.06"],
            ["Temporal style", "25.68", "25.89", "25.75", "+0.07"],
            ["Overall consistency", "22.96", "23.36", "23.37", "+0.41"],
            ["Human action", "92.00", "100.00", "100.00", "+8.00"],
            ["Temporal flickering", "99.74", "98.79", "98.81", "-0.93"],
            ["Motion smoothness", "98.63", "98.66", "98.55", "-0.08"],
            ["Dynamic degree", "93.33", "80.00", "93.33", "0.00"],
            ["Appearance style", "19.34", "19.44", "18.83", "-0.51"],
        ],
        highlights={(0, 4), (1, 4), (4, 4), (5, 4), (7, 4), (8, 4)},
        font_size=7.5,
    )

    document.add_heading("质量结论", level=2)
    add_bullet(document, "基础权重下 HSA+CAG 几乎保持 dense 质量：Total 82.80，仅下降 0.03；SLA 与混合路由未经适配时均下降约 5 分。", key=True)
    add_bullet(document, "200-step 微调可恢复部分稀疏质量，但匹配路由相对各自 dense 仍低 3.30（SLA）和 3.12（Hybrid），尚未达到质量等价。", key=True)
    add_bullet(document, "交叉实验显示路由专门化：SLA 权重更适合 SLA 路由，Hybrid 权重更适合 Hybrid 路由，交互效应为 1.88 分。")
    add_bullet(document, "主要残差集中在主体/背景一致性、对象类别、多对象与空间关系；5% 子集对离散维度较敏感，正式结论应补充全量 VBench 和重复种子。")

    document.add_heading("数据来源与限制", level=2)
    add_body(document, "性能原始记录与计算：docs/experiments/sparse_attention_performance_analysis.md")
    add_body(document, "质量原始记录与计算：docs/experiments/sparse_attention_quality_analysis.md")
    add_body(document, "本报告保留关键原始值和派生值。最新服务器运行的完整 manifest 未随本地仓库提供，因此跨运行对比应在发布前再次核对提示词、seed、SP/DP、checkpoint 与采样参数。")


def main():
    document = Document()
    configure_document(document)
    add_title(document)
    add_performance_section(document)
    add_quality_section(document)
    document.core_properties.title = "LongLive2.0 稀疏注意力实验结果"
    document.core_properties.subject = "性能数据与 VBench 质量数据汇总"
    document.core_properties.author = "LongLive-oneday"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
