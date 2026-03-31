---
name: audio-analyzer
description: "Audio analysis + lyrics transcription + LLM music producer synthesis + original song creation for Suno AI. Analyzes audio files (MP3/WAV/FLAC/AAC/OGG/M4A) returning key/mode/chords/bassline/drums/EQ/LUFS/BPM/melody/structure, auto vocal transcription with chorus detection, deep producer-perspective synthesis integrating audio features and lyric themes, then creates a brand-new original song (title + full lyrics + Suno style prompt ≤200 chars) imitating the reference track's style and production DNA. Keywords: 分析音频, suno prompt, 歌词识别, 仿写歌曲, 创作歌词, 模仿创作, 编曲参考, BPM, 和弦进行, 调性, bassline, audio analysis, lyrics transcription, imitation composition, original song creation."
version: "5.0"
changelog: "v5 — Song name detection from filename + web knowledge retrieval (genre articles, reviews, Wikipedia, producer interviews) fused into final Suno prompt for high-fidelity style replication. v4 — Multi-signal genre scoring system, genre-aware bass/drum tag selection, modal mood cross-validated with genre context, LLM synthesis hint."
---

# Audio Analyzer v5 — Production Analysis + Web Knowledge Fusion + Original Song for Suno

Full pipeline: audio feature extraction → **song identity detection → web knowledge retrieval** → vocal transcription → music producer deep analysis → original song creation → Suno prompt.

---

## Pipeline Overview

```
Step 0  Detect song identity  →  filename → song name + artist (if named)
Step 1  Run analysis script   →  raw JSON (audio features + whisper lyrics)
Step 1b Web knowledge fetch   →  genre articles, reviews, Wikipedia, producer info
Step 2  Present analysis report  →  structured sections ①–⑫
Step 3  LLM synthesis         →  fuse audio data + web knowledge, style tag validation
Step 4  Original song creation  →  new title + full lyrics + final Suno prompt
```

Steps 1 is script-driven. **Steps 0, 1b, 3–4 are Claude's work** — web knowledge fusion is the v5 core upgrade.

---

## Step 0 — Song Identity Detection（歌曲身份识别）

**在运行脚本之前，先从文件名中提取歌曲信息。**

```python
import os
filename = os.path.splitext(os.path.basename(file_path))[0]
# filename 就是用户起的歌名，例如："Blinding Lights" 或 "The Weeknd - Blinding Lights"
```

### 解析规则

| 文件名格式 | 解析结果 |
|---|---|
| `歌手名 - 歌名.mp3` | artist="歌手名", title="歌名" |
| `歌名.mp3` | title="歌名", artist=未知 |
| `01 歌名.mp3` | 去掉数字前缀，title="歌名" |
| `随机字符.mp3` / `recording.mp3` | 无法识别 → 跳过 Step 1b |

**识别结论（Step 0 输出）：**
```
🎵 识别到歌曲：[歌名] — [歌手]（如有）
   将在 Step 1b 搜索该曲目的风格资料
```
若文件名无法识别为歌名，直接标注"⚠️ 文件名无法识别为歌名，跳过网络查询，仅使用音频数据"。

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

## Step 1b — Web Knowledge Retrieval（网络知识检索）🌐 v5 新增

**脚本运行后，立即并行发起网络搜索，为 Step 3 风格融合做准备。**

仅当 Step 0 成功识别歌名时执行此步骤。

### 搜索策略

使用 agent-browser / web_fetch，依次抓取以下来源：

#### 优先级 1：Wikipedia / 百科（最权威）
```
搜索词："{歌名} {歌手} wikipedia"
目标：风格标签、专辑信息、制作人、影响力来源
提取字段：
  - genre（流派标签，Wikipedia 往往非常精准）
  - recorded at / produced by（制作背景）
  - influences / inspiration（影响来源）
  - chart performance（商业定位参考）
```

#### 优先级 2：乐评 / 专辑评测
```
搜索词："{歌名} {歌手} review production analysis"
         "{歌名} {歌手} 乐评 制作分析"
目标：专业乐评人对编曲、音色、情绪的描述词
提取字段：
  - 对具体乐器/音色的描述（e.g., "pulsing synth bass", "slap reverb snare"）
  - 情绪/氛围描述词（e.g., "nostalgic melancholy", "euphoric rush"）
  - 与哪些艺人/作品风格相近的比较描述
```

#### 优先级 3：制作人信息 / 制作技巧
```
搜索词："{歌名} {歌手} producer interview beat breakdown"
         "{歌手} {专辑名} production technique"
目标：具体制作细节（synth型号、鼓机、效果器、采样来源等）
提取字段：
  - 使用的合成器/乐器型号（e.g., "Juno-106 synth", "Roland TR-808"）
  - 制作手法（e.g., "pitched up vocal sample", "side-chain compression"）
  - BPM/调性确认（可与脚本结果交叉验证）
```

#### 优先级 4：歌词解析 / 主题分析（有歌词时）
```
搜索词："{歌名} {歌手} lyrics meaning analysis"
         "{歌名} {歌手} 歌词解析 主题"
目标：歌词主题、意象系统、叙事结构
提取字段：
  - 核心主题/意象
  - 歌词写作风格描述
  - 情感弧线
```

### 检索结果整理格式

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 网络知识检索结果：[歌名] — [歌手]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 来源 1 — Wikipedia/百科
   流派标签：[精确标签，如 "synth-pop, new wave, R&B"]
   制作人：[姓名]
   录制年份/专辑：[信息]
   关键描述：[1-3句最有价值的描述]

📝 来源 2 — 乐评
   乐评来源：[网站名]
   音色描述：[具体描述词，直接可用于 prompt]
   情绪描述：[具体词汇]

🎚️ 来源 3 — 制作信息
   使用器材/合成器：[型号列表]
   制作手法亮点：[简述]
   BPM/调性确认：[与脚本结果对比]

🎤 来源 4 — 歌词主题（有歌词时）
   核心主题：[关键词]
   写作风格：[描述]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏷️ 可用于 Suno Prompt 的高价值词汇提炼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
从以上来源提炼出最适合 Suno Style 字段的词汇（来自人类描述，非脚本生成）：

Genre: [精确流派，直接来自 Wikipedia]
Sound descriptors: [合成器/乐器词汇，如 "Juno-106 synth", "slap reverb"]
Mood: [精确情绪词，来自乐评]
Production style: [制作手法词汇]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 搜索失败处理

若搜索无结果或被封锁：
- 直接标注"⚠️ 网络查询未获有效结果，仅使用音频分析数据"
- 继续 Step 2，不影响后续流程

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

### Step 3a — Web Knowledge + Audio Data Fusion（知识融合 & 风格标签精修）🌐 v5 核心

**v5 新增：将 Step 1b 的网络检索结果与脚本音频数据三角互证，生成最终 Suno Style Prompt。**

#### 融合优先级规则

```
优先级 1（最高）：Wikipedia / 百科的 genre 标签
  → 这是人类音乐学家定义的精确流派，直接采用，不被脚本结果覆盖
  → 例：Wikipedia 写 "synth-pop, new wave" → prompt 直接用这两个词

优先级 2：乐评中的音色/质感描述词
  → 这些是脚本完全无法检测的维度（合成器型号、效果器风格）
  → 例：乐评写 "warm Juno-106 pad" → 直接加入 prompt
  → 例：乐评写 "slap-back echo reverb" → 加入 prompt

优先级 3：制作人访谈中的器材/技法信息
  → 具体到"用了 TR-808 鼓机"这类信息对 Suno 命中率极高
  → 例："Roland TR-808 drums" / "sampled from soul vinyl"

优先级 4：脚本音频分析数据（用于校验或填补空缺）
  → 当网络信息不足时，用脚本数据补充
  → 用于确认/纠正：BPM、调性、能量结构
  → 不直接覆盖优先级 1-3 的标签
```

#### 三角互证校验

```
1. BPM 交叉验证：
   脚本测出的 BPM vs 制作人/Wikipedia 说的 BPM
   若相差 >5，以制作人说法为准（可能是半速测量问题）

2. 流派交叉验证：
   脚本 genre_scores 第一名 vs Wikipedia genre
   若不一致 → 以 Wikipedia 为准，检查脚本评分是否有哪个维度误判

3. 情绪词交叉验证：
   脚本 modal_flavor 推断的情绪 vs 乐评情绪描述词
   若一致 → 保留脚本词；若不一致 → 以乐评词为准（人耳>算法）

4. Bass/鼓型验证：
   脚本 drum pattern + 制作人"用了 TR-808"
   结合两者输出最精确的鼓型标签
```

#### 最终 Suno Style Prompt 生成

```
融合后的 Prompt 结构（按信息来源标注）：

[Wikipedia Genre] + [乐评音色词] + [脚本 mood] + [制作技法词] + [脚本节奏/结构词]

示例：
原曲：The Weeknd - Blinding Lights
Wikipedia genre: "synth-pop, new wave, R&B"
乐评音色: "pulsing synth arpeggios", "gated reverb drums"
脚本推断: modal_flavor=Aeolian, BPM=171, four-on-the-floor kick

融合 prompt: "synth-pop, new wave, pulsing synth arpeggios, gated reverb drums, melancholic, driving, four-on-the-floor"
字符数：91/120 ✅
```

**输出格式：**
```
🔧 风格标签融合校正：
   脚本原始：[rule_based_prompt]
   网络补充词：[来自 Wikipedia/乐评/制作人的关键词]
   融合后（最终版）：[final prompt]
   数据来源说明：[哪些词来自哪个来源]
```

#### 仅有音频数据时的校正（无网络结果）

```
1. 查看 genre_scores_top5：得分最高的是否合理？
2. 验证 mood：centroid+crest+harmonic_ratio 三角验证
3. 验证 bass 标签：808 仅限 hip-hop 语境；DnB sub ≠ 808
4. 验证 drum 标签：trap hi-hats 仅限 hihat≥4 AND hip-hop 语境
5. 补充人耳推断词：Rhodes / analog synth / vocal chops / distorted guitar 等
```

输出：`✅ 仅用音频数据，校正完毕` 或 `🔧 修改说明：[改了什么、为什么]`

---

### Step 3b — 综合分析总结

### Role Setting
> 你现在是一位拥有20年经验的专业音乐制作人和词曲作者，精通电子音乐、流行音乐制作，曾参与多张商业专辑。你刚刚完成了对这首参考曲目的全面技术分析和资料查阅，现在需要综合音频数据与人类音乐知识做一次深度总结，为创作一首风格相仿的新歌曲做准备。

### Synthesis Output — Present this section after the analysis report:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎼 制作人视角：综合分析总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【整体定位】
用1-2句话定性这首歌：风格流派（结合网络资料修正）、目标听众、情绪定位、场景
如有网络来源：注明"来源：Wikipedia / [乐评网站名]"

【音乐DNA提炼】
- 和声骨架：核心进行的调式特征，色彩变化规律
- 节奏性格：律动感、鼓型特征、groove核心（如有器材信息则注明：如"TR-808驱动"）
- 声场设计：低频策略、空间感、动态处理风格（如乐评有描述则直接引用）
- 制作手法：最显著的3个制作特征（优先用网络资料确认的信息）

【歌词主题分析】
- 核心意象/关键词（Whisper识别 + 网络歌词分析交叉验证）
- 情感基调
- 歌词结构特点（信息密度、重复规律、叙事方式）
- 副歌核心句

【氛围情绪图谱】
用3-5个关键词描述（优先使用乐评中出现过的原话，用""引用）

【仿作创作方向】
新歌应该继承的要素：
- 保留：[具体列出，标注哪些来自网络信息、哪些来自脚本分析]
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

## Suno Prompt 自动生成规则（脚本内置 v4）

脚本 `suno_prompt` 字段自动从分析数据提取基础标签，Claude 在 Step 3a 中需要校正，在 Step 4 中最终使用。

### 字符限制

| 链路 | 字段 | 字符限制 |
|---|---|---|
| A（纯音乐）| Style only | ≤ 200 字符 |
| B（有歌词）| Style of Music | ≤ **120 字符**（严格） |
| B（有歌词）| Lyrics | ≤ 3000 字符 |

### 优先级顺序（高→低）

| 优先级 | 维度 | v4 逻辑 |
|---|---|---|
| 1 | Genre（最多2个）| 多信号评分系统，12+流派，细分优先于通用 |
| 2 | Mood | `modal_flavor` × Genre 联合推断（EDM 语境下 Phrygian→dark and ominous，非 brooding）|
| 3 | Bass | Genre-aware：Hip-hop→808 bass；DnB→rolling bass；通用→heavy sub bass |
| 4 | Drums | Pattern-aware：Trap→trap hi-hats；DnB→breakbeat drums；EDM→four-on-the-floor |
| 5 | Tempo | BPM 细分6档（downtempo/mid-tempo/uptempo/driving/high-energy/frenetic） |
| 6 | Structure | 能量谷底→高潮≥8dB → build and drop |
| 7 | Lyric context | 歌词情绪 / instrumental 标签 |
| 8 | Texture | 最后填空：atmospheric/wide dynamics/compressed |
| ~~9~~ | ~~Language~~ | ~~链路B时删除，Lyrics已体现~~ |

### 流派评分系统（v4 新增）

脚本对12+流派进行多维度打分，`production_style.genre_scores` 字段记录各流派得分。**Claude 在 Step 3a 中应查看这个字段，判断规则系统是否选对了主流派。** 若得分第一的流派听感不符，应以得分第二或第三的流派为准，或结合听感手动指定。

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
| **流派识别** | **★★★★★** | **v5：Wikipedia genre（最权威）+ v4 多信号评分双保险** |
| **风格标签** | **★★★★★** | **v5：Web知识融合（Wikipedia+乐评+制作人访谈）× 音频数据三角互证** |
| LLM综合分析 | ★★★★ | Claude音乐制作人视角综合，质量取决于输入数据完整度 |
| 仿作歌词 | ★★★★ | Claude创作，风格仿照但内容100%原创 |
| **Suno Prompt** | **★★★★★** | **v5：人类音乐知识（Wikipedia/乐评/制作人）+ 脚本数据 + Claude 三角融合** |

---

## 依赖安装

```bash
pip install librosa soundfile scipy faster-whisper
# ffprobe 已预装（ffmpeg 套件）
# 首次运行whisper会下载small模型（~245MB），缓存于 /tmp/whisper_models
```
