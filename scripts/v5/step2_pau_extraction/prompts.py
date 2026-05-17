from __future__ import annotations

import json
from typing import List, Optional

SYSTEM_PROMPT = """你是政策文本PAU分析专家。将输入文本分解为结构化PAU(Policy Action Unit)。

## PAU结构定义
- M: 程度词（加快、持续、进一步）→ 记录但不影响分类
- A: 动词 + A_type（direction|substantive|operational|intent）
- O: 政策对象 注意保持固定搭配完整
- S_scope: 范围限定（地理/行业）→ 国际、航空、枢纽
- S_focus: 战略方向 → 绿色、智慧、数字化
- S_stage: 阶段特征 → 精品、示范、试点
- S_type: 技术子类 → 全货机、冷链、定期

## 核心规则
1. 【固定搭配】以下词组不可拆分，整体作为O：
   航空货运、航线网络、机场货站、多式联运、空空中转、跨境电商
2. 【S分类】
   - 地理词(国际/亚洲)、行业词(航空/枢纽) → S_scope
   - 绿色/智慧/数字化 → S_focus（战略方向，非废话！）
   - 精品/示范/试点 → S_stage（发展阶段）
   - 全货机/冷链/定期 → S_type
   - 高效/全面/积极 → 丢弃（噪声词）
3. 【T级预判】
   - T1=领域(体系/网络) T2=方向 T3=实体 T4=工具
   - S_scope每增1个，下沉1级；S_focus/S_type下沉0.5级
4. 【Leaf判定】O为具体实体+A为substantive/operational → is_leaf_candidate=true

## 输出格式（严格JSON）
{"pau_list":[{"pau_id":"PAU_001","M":"","A":"动词","A_type":"类型","O":"对象","S_scope":[],"S_focus":[],"S_stage":[],"S_type":[],"pau_final":"标准化表述","t_level_base":"T1-4","t_level_adjusted":"T1-4","is_leaf_candidate":false,"leaf_reason":"理由"}]}
"""

FEW_SHOT_EXAMPLES = [
    {
        "input": "构建全球航空货运网络",
        "output": {
            "pau_list": [
                {
                    "pau_id": "PAU_001",
                    "M": "",
                    "A": "构建",
                    "A_type": "substantive",
                    "O": "航空货运网络",
                    "S_scope": ["全球"],
                    "S_focus": [],
                    "S_stage": [],
                    "S_type": [],
                    "pau_final": "全球航空货运网络",
                    "t_level_base": "T1",
                    "t_level_adjusted": "T2",
                    "is_leaf_candidate": False,
                    "leaf_reason": "O为网络级抽象",
                }
            ]
        },
        "note": "错误做法是O=网络,S=航空货运。航空货运是领域固定词，不可拆。",
    },
    {
        "input": "提升枢纽智慧绿色水平",
        "output": {
            "pau_list": [
                {
                    "pau_id": "PAU_001",
                    "M": "",
                    "A": "提升",
                    "A_type": "direction",
                    "O": "水平",
                    "S_scope": ["枢纽"],
                    "S_focus": ["智慧"],
                    "S_stage": [],
                    "S_type": [],
                    "pau_final": "枢纽智慧水平",
                    "t_level_base": "T2",
                    "t_level_adjusted": "T3",
                    "is_leaf_candidate": False,
                    "leaf_reason": "A为direction",
                },
                {
                    "pau_id": "PAU_002",
                    "M": "",
                    "A": "提升",
                    "A_type": "direction",
                    "O": "水平",
                    "S_scope": ["枢纽"],
                    "S_focus": ["绿色"],
                    "S_stage": [],
                    "S_type": [],
                    "pau_final": "枢纽绿色水平",
                    "t_level_base": "T2",
                    "t_level_adjusted": "T3",
                    "is_leaf_candidate": False,
                    "leaf_reason": "A为direction",
                },
            ]
        },
        "note": "智慧、绿色是战略方向(S_focus)，代表不同政策分支，必须拆分为两个PAU。",
    },
    {
        "input": "打造精品洲际货运航线",
        "output": {
            "pau_list": [
                {
                    "pau_id": "PAU_001",
                    "M": "",
                    "A": "打造",
                    "A_type": "substantive",
                    "O": "货运航线",
                    "S_scope": ["洲际"],
                    "S_focus": [],
                    "S_stage": ["精品"],
                    "S_type": [],
                    "pau_final": "精品洲际货运航线",
                    "t_level_base": "T3",
                    "t_level_adjusted": "T4",
                    "is_leaf_candidate": True,
                    "leaf_reason": "O为具体实体,A为substantive",
                }
            ]
        },
        "note": "精品是阶段特征(S_stage)，体现发达地区的成熟度，与欠发达地区的新开航线形成对比。",
    },
]


class PromptManager:
    def __init__(self) -> None:
        self.system_prompt = SYSTEM_PROMPT
        self.examples = FEW_SHOT_EXAMPLES

    def render_user_prompt(
        self,
        text: str,
        h_level: str,
        context_before: Optional[List[str]] = None,
        context_after: Optional[List[str]] = None,
        include_examples: bool = True,
    ) -> str:
        parts: List[str] = []

        if include_examples and self.examples:
            parts.append("## 参考示例")
            for ex in self.examples:
                parts.append(f"输入: {ex['input']}")
                parts.append(f"输出: {json.dumps(ex['output'], ensure_ascii=False)}")
                parts.append(f"要点: {ex['note']}")
                parts.append("")

        if self._is_hierarchical_context(context_before):
            parts.append("## 父级标题")
            for ctx in context_before or []:
                parts.append(ctx.strip())
            parts.append("")
        else:
            if context_before:
                parts.append("## 上文（最多2条）")
                for ctx in context_before[-2:]:
                    parts.append(f"- {self._shorten(ctx)}")
                parts.append("")

        parts.append("## 待分析文本")
        parts.append(f"【层级】{h_level}")
        parts.append(f"【内容】{text}")
        parts.append("")

        if context_after and not self._is_hierarchical_context(context_before):
            parts.append("## 下文（最多2条）")
            for ctx in context_after[:2]:
                parts.append(f"- {self._shorten(ctx)}")
            parts.append("")

        parts.append("请输出严格JSON格式的PAU分解结果：")
        return "\n".join(parts)

    def get_messages(
        self,
        text: str,
        h_level: str,
        context_before: Optional[List[str]] = None,
        context_after: Optional[List[str]] = None,
        include_examples: bool = True,
    ) -> List[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.render_user_prompt(
                    text,
                    h_level,
                    context_before=context_before,
                    context_after=context_after,
                    include_examples=include_examples,
                ),
            },
        ]

    @staticmethod
    def _shorten(text: str, limit: int = 120) -> str:
        cleaned = " ".join(text.strip().split())
        return cleaned if len(cleaned) <= limit else cleaned[:limit] + "..."

    @staticmethod
    def _is_hierarchical_context(context_before: Optional[List[str]]) -> bool:
        if not context_before:
            return False
        for item in context_before:
            stripped = item.strip()
            if stripped.startswith("【H1】") or stripped.startswith("【H2】"):
                return True
        return False
