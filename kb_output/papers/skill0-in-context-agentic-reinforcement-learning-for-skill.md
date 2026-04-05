---
title: "SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization"
authors:
  - Zhengxi Lu
  - Zhiyuan Yao
  - Jinyang Wu
  - Chengcheng Han
  - Qi Gu
  - Xunliang Cai
  - Weiming Lu
  - Jun Xiao
  - Yueting Zhuang
  - Yongliang Shen
tags:
  - agentic reinforcement learning
  - skill internalization
  - curriculum learning
  - LLM agent training
  - in-context learning
  - tool use
  - multi-turn agents
  - visual context compression
  - ALFWorld
  - search-augmented QA
  - zero-shot generalization
  - GRPO
  - agent memory
url: https://arxiv.org/abs/2604.02268
created: 2026-04-05
engaged: true
insightful: true
---
# SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization

## Summary
- SKILL0 trains LLM agents to internalize procedural skills into model weights via a progressive curriculum that starts with full skill context and systematically withdraws it, achieving zero-shot inference with no runtime skill retrieval.
- The key mechanism is In-Context Reinforcement Learning (ICRL): skills are provided as visual context during training rollouts but removed entirely at inference, with a Dynamic Curriculum that evaluates each skill file's on-policy helpfulness and drops skills only when the policy no longer benefits from them.
- Results on ALFWorld (+9.7% over AgentOCR) and Search-QA (+6.6%) show SKILL0 matches or beats skill-augmented methods (SkillRL) while using 5× fewer tokens per step (<0.5k vs. 2.21k), demonstrating that internalization can be more efficient than retrieval-augmented inference.

## Questions

## Refs
- Shao et al. 2024 — DeepSeekMath / GRPO: Group Relative Policy Optimization, the base RL algorithm (arXiv:2402.03300)
- Xia et al. 2026 — SkillRL: Recursive skill-augmented RL, closest prior work and source of SkillBank initialization (arXiv:2602.08234)
- Feng et al. 2026 — AgentOCR: Visual context compression via optical self-compression, source of the rendering approach and composite reward design (arXiv:2601.04786)
- Shridhar et al. 2020 — ALFWorld: Primary evaluation benchmark for embodied text-based agents (arXiv:2010.03768)
- Jin et al. 2025 — Search-R1: Training LLMs to reason and leverage search engines with RL, source of Search-QA setup (arXiv:2503.09516)
- Yao et al. 2022 — ReAct: Synergizing reasoning and acting in language models, foundational agent baseline
- Shinn et al. 2024 — Reflexion: Language agents with verbal reinforcement learning (arXiv:2303.11366)
- Zhao et al. 2024 — ExpeL: LLM agents as experiential learners, foundational memory-based agent work
- Guo et al. 2025 — DeepSeek-R1: Incentivizing reasoning capability in LLMs via RL (arXiv:2501.12948)
- Yuan et al. 2025 — From f(x) and g(x) to f(g(x)): LLMs learn new skills in RL by composing old ones (arXiv:2509.25123)
- Anderson 1982 — Acquisition of cognitive skill: Psychological theory motivating the explicit/internalized skill progression
- Feng et al. 2025 — GiGPO: Group-in-group policy optimization for LLM agent training (arXiv:2505.10978)
- Wang et al. 2023 — Voyager: Open-ended embodied agent with LLMs, early skill-based agent work (arXiv:2305.16291)

## Notes
- Watchlist match: HIGH INTEREST match: Agent architectures (planning, reasoning, tool use) — directly advances agent training methodology. MODERATE match: Foundation model post-training — novel RL curriculum for skill internalization. The safety angle (internalized vs. inspectable skills) is an implicit concern worth flagging for safety researchers.

### Triage
7 — Solid contribution to agentic RL and agent training methodology. The core idea of using skills as transient training scaffolding that gets progressively withdrawn is novel and practically motivated. Not from a top safety lab, and the safety implications are indirect, but the work directly advances autonomous agent capabilities and addresses a real failure mode (inference-time context dependence). Worth reading for anyone working on agent training, tool use, or reducing inference-time overhead.

### Motivation
The prevailing paradigm for extending LLM agent capabilities is inference-time skill augmentation: retrieve relevant skills from a bank and inject them into the context at each step. This has three fundamental problems: (1) retrieval noise introduces irrelevant or misleading guidance; (2) injected skill content imposes compounding token overhead across multi-turn interactions; (3) most critically, the model never actually *learns* the skills — competence resides in the context, not the model. The authors draw an analogy to human cognitive skill acquisition (Anderson, 1982): humans move from explicit instruction to autonomous internalized execution. Inference-time augmentation permanently anchors agents in the first stage. The paper asks whether RL can drive the transition to the second stage.

### Hypothesis
Skills can be internalized into model parameters via a training-time curriculum that provides structured skill guidance during RL rollouts but progressively withdraws it, ultimately enabling zero-shot autonomous behavior without any runtime skill retrieval. The hypothesis is that RL optimization, when given the right scaffolding (skills) and the right pressure (progressive removal), will consolidate effective strategies as intrinsic policy rather than context-dependent behavior.

### Methodology
**Architecture:** Built on Qwen2.5-VL (3B and 7B), a vision-language model. Interaction history and skill files are rendered as compact RGB images (visual context), encoded by the vision encoder, dramatically reducing token overhead. The agent also self-generates a compression ratio at each step alongside its action.

**Skill Organization:** A hierarchical SkillBank with general skills (cross-task) and task-specific skills, organized as Markdown files by category (e.g., `skills/ALFWorld/clean.md`). Skills are initialized from SkillRL's bank.

**In-Context RL (ICRL):** Standard GRPO-style RL (group rollouts, advantage normalization, clipped importance sampling, KL penalty) with a composite reward: task success + log(compression ratio) when successful. Skills are provided as visual context during training rollouts.

**Dynamic Curriculum (3 stages):** (a) Offline Relevance-Driven Skill Grouping: each skill file is paired with a dedicated validation sub-task based on domain/category alignment. (b) Online Helpfulness-Driven Curriculum: every d=10 training steps, each skill file's helpfulness Δk is computed as accuracy with skill minus accuracy without skill on its paired sub-task. Skills with Δk ≤ 0 are filtered; remaining are ranked and top-M(s) are selected. Budget M(s) decays linearly from N to 0 across NS=3 stages. Final stage: no skills at all.

**Evaluation:** ALFWorld (3,827 household task instances, 6 categories) and Search-based QA (7 datasets: NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle). Training: 180 steps on 4 H800 GPUs.

### Results
**ALFWorld (3B):** SKILL0 achieves 87.9% average success rate vs. AgentOCR 78.2% (+9.7%), GRPO 79.9%, SkillRL† 82.4% (skill-augmented at inference). Token cost: 0.38k/step vs. SkillRL's 2.21k (5.8× reduction).

**ALFWorld (7B):** SKILL0 achieves 89.8% vs. AgentOCR 81.2% (+8.6%), GRPO 81.8%, SkillRL† 89.9% (essentially tied, but SKILL0 uses no inference-time skills). Beats GPT-4o (48.0%) and Gemini-2.5-Pro (60.3%) by large margins.

**Search-QA (3B):** 40.8% avg vs. AgentOCR 34.2% (+6.6%), GRPO 36.4%, SkillRL† 38.9%. Token cost: 0.18k/step vs. SkillRL's 0.87k (4.8× reduction). Particularly strong on Bamboogle (63.7%), an out-of-domain multi-hop dataset.

**Ablations:** Fixed full skill budget collapses by -12.3 to -13.3 points when skills are removed at inference. SKILL0's curriculum achieves +1.6 gain when skills are removed (better without skills than with). Removing the Rank step causes -13.7 point collapse. Removing Filter causes -2.7 drop. Validation interval d=10 is optimal.

**Training dynamics:** Helpfulness shows a characteristic rise-then-fall pattern across all sub-tasks, empirically validating the internalization mechanism. SKILL0 continues improving throughout training while GRPO and SkillRL plateau early.

### Interpretation
The paper makes a genuinely interesting conceptual contribution: reframing skill augmentation not as an inference-time mechanism but as a training-time scaffold. The analogy to human cognitive skill acquisition (Anderson's ACT* theory) is apt and well-motivated. The Dynamic Curriculum is the key technical innovation — the helpfulness-driven adaptive withdrawal is more principled than fixed annealing schedules and avoids the failure mode of premature skill removal.

**What's surprising:** The finding that SKILL0 performs *better* without skills at inference than with them (+1.6%) is striking. This suggests the curriculum not only internalizes skills but also eliminates a form of context-dependence that would otherwise hurt performance. This is reminiscent of the training-inference distribution gap problem in RAG systems.

**Questionable assumptions:** (1) The theoretical analysis (Appendix A) assumes a locally additive utility function J(S) ≈ Σ Δk, which is a strong independence assumption that may not hold when skills interact. (2) The SkillBank is initialized from SkillRL's bank, which uses privileged validation trajectories — this is a non-trivial dependency that the paper acknowledges but doesn't fully address. (3) The visual rendering approach (OCR-style) is borrowed from AgentOCR and is a significant component of the token efficiency gains, somewhat conflating two separate contributions.

**Relation to D2Skill (KB):** There's a notable parallel with D2Skill (arXiv:2603.28716) in the KB, which also observed that skills partially internalize into policy weights during training. SKILL0 makes this the *explicit* training objective rather than a side effect, which is the key distinction. D2Skill uses paired rollouts for utility estimation; SKILL0 uses separate validation sub-tasks — both are principled approaches to the same problem.

### Context
This paper sits at the intersection of several active research threads: (1) agentic RL post-training (GRPO, Search-R1, AgentOCR), (2) skill-augmented agents (SkillRL, D2Skill, EvolveR), and (3) curriculum learning for RL. The closest prior work is SkillRL (Xia et al. 2026), which SKILL0 directly competes with and matches despite using no inference-time skills. The visual context compression approach builds on AgentOCR (Feng et al. 2026). The broader context is the emerging ecosystem of "agent skills" (Xu & Yan 2026) as a standard mechanism for extending LLM agent capabilities — SKILL0 challenges this paradigm by arguing for internalization over augmentation. The paper is from Zhejiang University and Meituan (a Chinese tech company), with the first author interning at Meituan. Not from a top safety lab, but the technical quality is solid.

### Limitations
1. **SkillBank dependency:** SKILL0 relies on the quality of the initial SkillBank (initialized from SkillRL), which itself requires privileged validation trajectories and an external LLM for skill generation. The paper doesn't address how to build the SkillBank from scratch. 2. **Domain re-partitioning:** The offline relevance-driven skill grouping requires re-partitioning when applied to new task domains, limiting plug-and-play applicability. 3. **Limited benchmark diversity:** Only two benchmark types (embodied household tasks and search-based QA). No evaluation on code generation, GUI automation, or other agent domains where skills are increasingly used. 4. **Additive utility assumption:** The theoretical justification assumes skill utilities are independent and additive, which may not hold in practice. 5. **Visual rendering dependency:** The token efficiency gains are partly attributable to the OCR-style visual rendering (borrowed from AgentOCR), not purely to skill internalization. These contributions are somewhat entangled. 6. **Short training:** Only 180 steps — unclear if the curriculum would remain stable with longer training or larger skill banks. 7. **No safety analysis:** The paper doesn't consider whether internalized skills could encode unsafe behaviors that are harder to inspect or remove than context-injected skills.

### Why it matters
**Relevance to AI Safety & Agents Research (Score: 7/10):** Directly relevant to agent architecture and training methodology. The skill internalization paradigm has implications for agent controllability: if skills are baked into weights rather than retrieved from an inspectable context, it becomes harder to audit or modify agent behavior at inference time — a potential safety concern not discussed by the authors. The curriculum learning approach is a clean example of training-time scaffolding that could generalize to other forms of knowledge transfer. The token efficiency gains are practically important for deployed agent systems. The paper is not from a priority organization (Anthropic, OpenAI, DeepMind) and doesn't directly address safety alignment, but the agent training methodology is directly in the high-interest zone.
