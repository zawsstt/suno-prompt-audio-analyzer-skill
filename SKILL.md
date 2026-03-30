---
name: audio-analyzer
description: "Audio analysis + lyrics transcription + LLM music producer synthesis + original song creation for Suno AI. Analyzes audio files (MP3/WAV/FLAC/AAC/OGG/M4A) returning key/mode/chords/bassline/drums/EQ/LUFS/BPM/melody/structure, auto vocal transcription with chorus detection, deep producer-perspective synthesis integrating audio features and lyric themes, then creates a brand-new original song (title + full lyrics + Suno style prompt ≤200 chars) imitating the reference track's style and production DNA. Keywords: 分析音频, suno prompt, 歌词识别, 仿写歌曲, 创作歌词, 模仿创作, 编曲参考, BPM, 和弦进行, 调性, bassline, audio analysis, lyrics transcription, imitation composition, original song creation."
---

# Audio Analyzer v3 — Production Analysis + LLM Synthesis + Original Song for Suno

Full pipeline: audio feature extraction → vocal transcription → music producer deep analysis → original song creation → Suno prompt.

---

## Pipeline Overview

```
Step 1  Run analysis script  →  raw JSON (audio features + whisper lyrics)
Step 2  Present analysis report  →  structured sections ①–⑫
Step 3  LLM synthesis  →  music producer perspective, integrate audio + lyrics
Step 4  Original song creation  →  new title + full lyrics + Suno prompt ≤200 chars
```

Steps 1–2 are script-driven. **Steps 3–4 are Claude's creative work** — this is the core value of the skill.

---

## Step 1 — Run Analysis Script

### Install dependencies (first time only)
```bash
pip install librosa soundfile scipy faster-whisper --quiet
```

### Run
```bash
python3 ~/.openclaw/skills/audio-analyzer/scripts/analyze_audio.py <file_path>
```

### Download if URL
```bash
curl -sL -o /tmp/files/$(date +%s).wav "<URL>"
```

---

## Step 2 — Present Analysis Report

Parse the JSON and present all sections clearly. Use this structure:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎛️ PRODUCTION QUICK-START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BPM / Key / Time Sig / Swing / LUFS / Dynamic Range

① 响度 & 动态
RMS / LUFS / 峰值 / Crest Factor / 动态范围 / Headroom + 压缩提示

② 频段能量（EQ参考）
7-band 条形图 + % + dB；注明低频/高频导向

③ 节奏 & 律动
BPM / Swing类型 / 拍号 / 前16拍时间轴

④ 鼓组 Pattern（前8小节）
Kick / Snare / HiHat 时间点 + 每小节密度

⑤ Bassline 音符序列
最常出现低音根音表格 + 开头行进序列 + MIDI建议

⑥ 和弦进行
核心loop（带罗马数字）/ 全曲和弦 / 最高频跳转 / 色彩变化点

⑦ 调性 & 音阶
主调 / 调式色彩（Phrygian/Dorian/etc.）/ 相对调 / 色度能量分布

⑧ 调性时间轴（每15秒）
时间段 → 调性 / 置信度 / 变化标记

⑨ 旋律轮廓
音域范围 / 主导旋律音 / 开头走向序列

⑩ 能量弧度 & 段落结构
每5秒 RMS+Flux 时间轴，高亮drop/peak
推断段落：Intro / Verse / Build / Chorus / Breakdown / Drop / Outro

⑪ 音色 & 频谱特征
质心 / 带宽 / 滚降 / 谐波比 / 亮度描述

⑫ 歌词识别（Whisper）
语言 / 人声密度 / 完整歌词时间轴
副歌候选行（重复次数最多的行）
歌词主题关键词
```

---

## Step 3 — LLM Music Producer Synthesis（核心环节）

**在输出分析报告之后，以专业音乐制作人身份进行一次综合梳理。**

### Role Setting
> 你现在是一位拥有20年经验的专业音乐制作人和词曲作者，精通电子音乐、流行音乐制作，曾参与多张商业专辑。你刚刚完成了对这首参考曲目的全面技术分析，现在需要用你的专业视角做一次深度总结，为创作一首风格相仿的新歌曲做准备。

### Synthesis Output — Present this section after the analysis report:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎼 制作人视角：综合分析总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【整体定位】
用1-2句话定性这首歌：风格流派、目标听众、情绪定位、场景

【音乐DNA提炼】
- 和声骨架：核心进行的调式特征，色彩变化规律
- 节奏性格：律动感、鼓型特征、groove核心
- 声场设计：低频策略、空间感、动态处理风格
- 制作手法：最显著的3个制作特征（如：breakdown蓄力、sub bass主导、直拍量化等）

【歌词主题分析】
- 核心意象/关键词（来自whisper识别）
- 情感基调
- 歌词结构特点（信息密度、重复规律、叙事方式）
- 副歌核心句

【氛围情绪图谱】
用3-5个关键词描述这首歌营造的情绪体验（如：压迫感、宿命感、爆发感、神秘感…）

【仿作创作方向】
新歌应该继承的要素：
- 保留：[具体列出]
- 变化：[主题/意象/情绪可以做哪些差异化]
```

---

## Step 4 — Original Song Creation（仿作新歌）

**基于综合分析，先判断原曲类型，走不同的创作链路。**

---

### 🔀 类型判断

| 条件 | 类型 | 创作链路 |
|---|---|---|
| `lyrics.has_lyrics == false` 或 `lyrics.has_lyrics` 不存在 | **纯音乐** | → 链路 A |
| `lyrics.has_lyrics == true` | **有歌词音乐** | → 链路 B |

---

### 链路 A — 纯音乐（Instrumental）

只需输出一个 Suno Style Prompt（≤200字符），不创作歌词。

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎵 仿作创作（纯音乐）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 参考曲风格定位：
   [制作人视角1句话总结]

🎚️ Suno Style Prompt（≤200字符，直接粘贴）：
   [style prompt]
   字符数：XX/200

微调建议：
   [2-3条可选追加词，如：更暗沉/更空灵/更激进]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 链路 B — 有歌词音乐

Suno 有两个独立输入框，字符限制不同，**必须分开输出**：

| 字段 | 限制 | 内容 |
|---|---|---|
| **Lyrics（歌词）** | ≤ 3000 字符 | 完整歌词，含 Suno 段落标签 |
| **Style of Music（风格）** | ≤ **120 字符** | 纯风格描述词，不含歌词内容 |

#### 创作原则

**核心：仿原曲歌词的"模子"，不套通用模板。**

在写新词之前，必须先从 `lyrics.timeline` 和 `lyrics.full_text_preview` 中提炼出原曲的歌词"模子"：

| 维度 | 需要观察的内容 |
|---|---|
| **行长节奏** | 每行音节数，短句/长句的混合规律 |
| **断句方式** | 句子是否跨行断开？有无故意留白的单行？|
| **押韵方案** | AABB / ABAB / 隔行押 / 不规则押？韵脚在哪个位置？|
| **用词语气** | 口语化（Yeah/Ain't/just）/ 正式 / AAVE / 诗意？|
| **意象风格** | 具体画面（"grip" "bottles"）还是抽象概念？|
| **段落结构** | 原曲实际有几个段落？Rap Bridge 有无？长短如何？|
| **叙事口吻** | 第一人称直诉？对话？内心独白？|

**提炼完"模子"后，按以下规则写新词：**
1. **行长和断句方式跟着原曲走**，不是跟着通用Suno模板走
2. **段落数量和顺序仿原曲**，原曲有Rap Bridge就写Rap Bridge，没有就不硬加
3. **用词语气和意象密度仿原曲**，原曲直白就直白，原曲隐晦就隐晦
4. **主题/意象/歌名完全原创**，内容不重复原曲，但"感觉"要像出自同一位词人
5. **Suno 段落标签**仅使用原曲出现过的结构，不硬套固定模板
6. **严禁自行添加原曲没有的段落**——原曲以副歌反复淡出结尾就不加 [Outro]，原曲无 [Breakdown] 就不写 [Breakdown]，以此类推
7. **结尾方式完全跟原曲**——原曲副歌收尾就副歌收尾，原曲单行收尾就单行收尾，不自作主张加"极简三行"或任何补充段落
6. **Style 字段严格≤120字符**，超限必须删减

#### Output Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎵 仿作新歌创作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 歌名：[原创歌名]
   概念：[一句话说明主题]

🎭 歌词"模子"提炼：
   行长：[X-X音节，短句为主/混合/长句为主]
   断句：[是否有跨行断句/刻意留白单行？]
   押韵：[押韵方案描述]
   语气：[口语/正式/AAVE/诗意？代表词汇举例]
   意象：[具体画面/抽象概念，举例]
   结构：[原曲实际段落顺序]

🎭 创作说明：
   [继承了哪些核心元素 / 在哪些维度做了差异化]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 INPUT 1 — Lyrics 字段（≤3000字符，直接粘贴）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[按原曲实际结构创作，段落标签仿原曲，不套固定模板]

字符数：XXXX/3000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎚️ INPUT 2 — Style of Music 字段（≤120字符，直接粘贴）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[style prompt，严格≤120字符，只含风格/情绪/乐器描述词]
字符数：XX/120

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 使用说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Suno → Create → Custom Mode
2. INPUT 1 歌词 → 粘贴到 Lyrics 框
3. INPUT 2 风格 → 粘贴到 Style of Music 框
4. Title 填入歌名，点击 Create

微调建议：[2-3条]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Suno Prompt 自动生成规则（脚本内置）

脚本 `suno_prompt` 字段自动从分析数据提取基础标签，Claude 在 Step 4 中需根据链路类型调整字符限制：

| 链路 | 字段 | 字符限制 |
|---|---|---|
| A（纯音乐）| Style only | ≤ 200 字符 |
| B（有歌词）| Style of Music | ≤ **120 字符**（严格） |
| B（有歌词）| Lyrics | ≤ 3000 字符 |

脚本输出的 `suno_prompt.prompt` 基于200字符预算生成，**链路B时 Claude 必须手动压缩至120字符**，优先保留 Genre + Mood + 最核心的1-2个乐器/制作特征词，删除 Language 标签（已在 Lyrics 字段体现）。

优先级顺序（高→低）：

| 优先级 | 维度 | 数据来源 |
|---|---|---|
| 1 | Genre（最多2个）| `likely_genres`，电子细分优先于pop |
| 2 | Mood | `modal_flavor` 映射（Phrygian→dark, Dorian→melancholic…）|
| 3 | Bass | `sub_bass_energy`（>15%→heavy sub bass）|
| 4 | Drums | kick/hihat密度（≥3.5/bar→four-on-the-floor）|
| 5 | Tempo | BPM范围描述（120-140→driving）|
| 6 | Structure | 能量谷底→高潮≥8dB → build and drop |
| 7 | Dynamics | crest factor（>14dB→wide dynamics）|
| ~~8~~ | ~~Language~~ | ~~链路B时删除，Lyrics已体现~~ |

---

## 精度说明

| 维度 | 精度 | 说明 |
|---|---|---|
| BPM | ★★★★ | ±1 BPM，拍号建议耳听确认 |
| 调性/调式 | ★★★ | KS统计算法，转调歌曲偏差较大 |
| 和弦 | ★★★ | 仅三和弦，无七和弦/延伸音 |
| Bassline | ★★★★ | pyin tracking，前90秒有效 |
| 鼓组 | ★★★ | 频段分离法，前8小节有效 |
| 歌词识别 | ★★★ | faster-whisper small，英语优；重混音时准确率下降 |
| LLM综合分析 | ★★★★ | Claude音乐制作人视角综合，质量取决于输入数据完整度 |
| 仿作歌词 | ★★★★ | Claude创作，风格仿照但内容100%原创 |
| Suno Prompt | ★★★★ | 脚本提取+Claude优化组合 |

---

## 依赖安装

```bash
pip install librosa soundfile scipy faster-whisper
# ffprobe 已预装（ffmpeg 套件）
# 首次运行whisper会下载small模型（~245MB），缓存于 /tmp/whisper_models
```
