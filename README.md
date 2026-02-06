\# AI-Scientist (SakanaAI) – Eli Lilly LNP Study



This repository contains the full output of running the \*\*SakanaAI AI-Scientist framework (AI-Scientist-v2)\*\* on the Eli Lilly research prompt:



> Investigating Non-covalent Interactions Between Oxidized Ionizable Lipids and siRNA in Lipid Nanoparticle Systems



\## AI Framework and Model

\- Framework: \*\*SakanaAI – AI-Scientist-v2\*\*

\- Language model used: \*\*gpt-4o-mini (lightweight model)\*\*



---



\## Input Prompt

The full research prompt provided by the instructor is documented in:
\- `input/eli_lilly_prompt.txt`

\- `idea.md`

\- `idea.json`



These files contain the exact problem description and constraints used to initiate the AI Scientist run.



---



\## Intermediate Steps (Chat / Reasoning Trace)

The interaction history, including prompts, intermediate reasoning steps, and model outputs, is recorded in:



\- `token\_tracker\_interactions.json`

\- `token\_tracker.json`

\- `logs/0-run/` (stage journals and summaries)

\- `logs/0-run/unified\_tree\_viz.html` (visualization of the agent workflow)



These files together represent the “chat” and intermediate steps of the AI framework.



---



\## Final Output

The generated research paper is provided as a PDF:



\- `2026-02-06\_07-43-32\_eli\_lilly\_lnp\_1\_attempt\_0\_1.pdf`



Earlier compilation attempts are also included for reference:

\- `2026-02-06\_07-43-32\_eli\_lilly\_lnp\_1\_attempt\_0\_0.pdf`



---



\## Notes

This run uses a lightweight language model and does not execute real all-atom molecular dynamics simulations. The results should be interpreted as an AI-generated research-style output rather than a fully physically validated MD study.

